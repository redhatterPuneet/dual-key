#!/usr/bin/env bash
# Prepares an OpenShift-on-AWS cluster for Dual-Key and deploys the operator.
#
#   1. adds 3 GPU worker nodes, one per GPU type:  g4dn.xlarge (T4), g5.xlarge (A10G), g6.xlarge (L4)
#   2. installs Node Feature Discovery and the NVIDIA GPU Operator, waits for GPUs to be ready
#   3. builds the Dual-Key image in the cluster and deploys the operator
#   4. creates a DualKeyCluster in discovery mode and prints the dashboard URL
#
# Run from the dualkey-operator folder, after logging in:
#   oc login https://api.<cluster>:6443 -u admin
#   ./scripts/setup-aws-gpus.sh            # everything
#   ./scripts/setup-aws-gpus.sh --nodes-only
#   ./scripts/setup-aws-gpus.sh --deploy-only
#
# Safe to re-run: every step checks what already exists.
set -euo pipefail

GPU_TYPES=(${GPU_TYPES:-g4dn.xlarge g5.xlarge g6.xlarge})
ROOT_VOLUME_GB=${ROOT_VOLUME_GB:-200}
NS=dualkey
STEP_NODES=1; STEP_DEPLOY=1
case "${1:-}" in
  --nodes-only) STEP_DEPLOY=0 ;;
  --deploy-only) STEP_NODES=0 ;;
  "") ;;
  *) echo "usage: $0 [--nodes-only|--deploy-only]"; exit 1 ;;
esac

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

wait_for() {  # wait_for <seconds> <description> <command...>
  local timeout=$1 what=$2; shift 2
  local start=$SECONDS
  until "$@" >/dev/null 2>&1; do
    (( SECONDS - start > timeout )) && return 1
    printf '.'; sleep 15
  done
  echo
}

command -v oc >/dev/null || die "oc not found in PATH"
command -v jq >/dev/null || die "jq not found (brew install jq)"
oc whoami >/dev/null 2>&1 || die "not logged in: run  oc login https://api.<cluster>:6443 -u admin  first"
[[ -f deploy/kustomization.yaml ]] || die "run this from the dualkey-operator folder"
info "logged in as $(oc whoami) on $(oc whoami --show-server)"
oc auth can-i '*' '*' --all-namespaces >/dev/null || die "cluster-admin rights are needed"

# ---------------------------------------------------------------- 1. GPU nodes
if (( STEP_NODES )); then
  say "1/4  Adding GPU nodes: ${GPU_TYPES[*]}"
  oc get infrastructure cluster -o jsonpath='{.status.platform}' | grep -qi aws || die "this cluster is not on AWS"
  INFRA_ID=$(oc get infrastructure cluster -o jsonpath='{.status.infrastructureName}')
  BASE=$(oc get machinesets -n openshift-machine-api -o json |
    jq -r '[.items[] | select((.spec.template.spec.providerSpec.value.instanceType // "") | test("^(g|p)[0-9]") | not)][0].metadata.name')
  [[ -n "$BASE" && "$BASE" != null ]] || die "no worker MachineSet found to copy"
  info "copying MachineSet $BASE (infra $INFRA_ID)"

  for TYPE in "${GPU_TYPES[@]}"; do
    SHORT=$(echo "$TYPE" | cut -d. -f1)               # g4dn / g5 / g6
    NAME="$INFRA_ID-gpu-$SHORT"
    if oc get machineset "$NAME" -n openshift-machine-api >/dev/null 2>&1; then
      info "$NAME already exists"; continue
    fi
    oc get machineset "$BASE" -n openshift-machine-api -o json | jq \
      --arg name "$NAME" --arg type "$TYPE" --arg short "$SHORT" --argjson vol "$ROOT_VOLUME_GB" '
      del(.status, .metadata.uid, .metadata.resourceVersion, .metadata.creationTimestamp,
          .metadata.generation, .metadata.annotations, .metadata.managedFields)
      | .metadata.name = $name
      | .spec.replicas = 1
      | .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.spec.metadata.labels = ((.spec.template.spec.metadata.labels // {})
          + {"dualkey.io/gpu-pool": "true", "dualkey.io/gpu-type": $short})
      | .spec.template.spec.taints = [{"key": "nvidia.com/gpu", "value": "true", "effect": "NoSchedule"}]
      | .spec.template.spec.providerSpec.value.instanceType = $type
      | .spec.template.spec.providerSpec.value.blockDevices[0].ebs.volumeSize = $vol
      ' | oc apply -f -
    info "created $NAME ($TYPE)"
  done

  say "     Waiting for the GPU machines to start (usually 5-10 minutes)"
  gpu_nodes_ready() {
    [[ $(oc get nodes -l dualkey.io/gpu-pool=true --no-headers 2>/dev/null | grep -c ' Ready') -ge ${#GPU_TYPES[@]} ]]
  }
  if ! wait_for 1500 "GPU nodes" gpu_nodes_ready; then
    echo
    oc get machines -n openshift-machine-api -o wide | grep -E "NAME|gpu-" || true
    oc get machines -n openshift-machine-api -o json |
      jq -r '.items[] | select(.metadata.name | test("gpu-")) | select(.status.errorMessage != null)
             | "   \(.metadata.name): \(.status.errorMessage)"' || true
    die "GPU nodes did not become Ready. A common cause on workshop accounts is a zero AWS quota for
       'Running On-Demand G and VT instances' or an instance type not offered in this zone.
       Fix the quota or set GPU_TYPES (e.g. GPU_TYPES=\"g5.xlarge g5.xlarge g6.xlarge\") and re-run."
  fi
  oc get nodes -l dualkey.io/gpu-pool=true -L dualkey.io/gpu-type,node.kubernetes.io/instance-type

  # ------------------------------------------------------------ 2. GPU operators
  say "2/4  Installing Node Feature Discovery and the NVIDIA GPU Operator"
  install_operator() {  # ns package source
    local ns=$1 pkg=$2 src=$3
    oc get ns "$ns" >/dev/null 2>&1 || oc create ns "$ns"
    oc get operatorgroup -n "$ns" -o name | grep -q . || cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata: {name: $pkg, namespace: $ns}
spec: {targetNamespaces: [$ns]}
EOF
    local channel
    channel=$(oc get packagemanifest "$pkg" -n openshift-marketplace -o jsonpath='{.status.defaultChannel}')
    [[ -n "$channel" ]] || die "package $pkg not found in OperatorHub"
    cat <<EOF | oc apply -f -
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata: {name: $pkg, namespace: $ns}
spec: {name: $pkg, channel: "$channel", source: $src, sourceNamespace: openshift-marketplace, installPlanApproval: Automatic}
EOF
    info "subscribed to $pkg ($channel) in $ns"
  }
  csv_ready() { oc get csv -n "$1" -o jsonpath='{.items[*].status.phase}' | grep -q Succeeded; }

  install_operator openshift-nfd nfd redhat-operators
  install_operator nvidia-gpu-operator gpu-operator-certified certified-operators
  printf '   waiting for NFD operator'; wait_for 900 "nfd" csv_ready openshift-nfd || die "NFD operator did not install"
  printf '   waiting for GPU operator'; wait_for 900 "gpu" csv_ready nvidia-gpu-operator || die "GPU operator did not install"

  if ! oc get nodefeaturediscovery -n openshift-nfd -o name | grep -q .; then
    oc get csv -n openshift-nfd -o json | jq '.items[0].metadata.annotations["alm-examples"] | fromjson
      | map(select(.kind == "NodeFeatureDiscovery"))[0]' | oc apply -n openshift-nfd -f -
    info "created NodeFeatureDiscovery"
  fi
  if ! oc get clusterpolicy -o name | grep -q .; then
    oc get csv -n nvidia-gpu-operator -o json | jq '.items[] | select(.metadata.name | test("gpu-operator"))
      | .metadata.annotations["alm-examples"] | fromjson | map(select(.kind == "ClusterPolicy"))[0]' | oc apply -f -
    info "created ClusterPolicy (drivers install on each GPU node: 10-15 minutes)"
  fi

  say "     Waiting until every GPU node advertises nvidia.com/gpu"
  gpus_allocatable() {
    local n
    n=$(oc get nodes -l dualkey.io/gpu-pool=true -o json |
      jq '[.items[] | select((.status.allocatable["nvidia.com/gpu"] // "0") | tonumber > 0)] | length')
    [[ "$n" -ge ${#GPU_TYPES[@]} ]]
  }
  wait_for 2400 "gpus" gpus_allocatable || {
    oc get pods -n nvidia-gpu-operator | grep -v Running || true
    die "GPUs not ready yet. Check: oc get pods -n nvidia-gpu-operator"; }
  oc get nodes -l dualkey.io/gpu-pool=true \
    -o custom-columns='NODE:.metadata.name,TYPE:.metadata.labels.node\.kubernetes\.io/instance-type,GPU:.status.allocatable.nvidia\.com/gpu,MODEL:.metadata.labels.nvidia\.com/gpu\.product'
fi

# ---------------------------------------------------------------- 3. operator
if (( STEP_DEPLOY )); then
  say "3/4  Deploying the Dual-Key operator"
  oc apply -k deploy/
  oc start-build dualkey -n "$NS" --from-dir=. --follow --wait
  oc rollout restart deployment/dualkey-operator -n "$NS"
  oc rollout status deployment/dualkey-operator -n "$NS" --timeout=300s

  # ------------------------------------------------------------ 4. DualKeyCluster
  say "4/4  Creating the DualKeyCluster (discovery mode)"
  oc apply -f deploy/samples/dualkeycluster-aws.yaml
  printf '   waiting for the dashboard'
  wait_for 600 "ui" oc rollout status deployment/dualkey-ui -n "$NS" --timeout=10s || info "dashboard still starting"
  URL=$(oc get route dualkey-dashboard -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)
  say "Done"
  info "Dashboard:      https://$URL"
  info "Groups:         oc get cg -n $NS"
  info "GPU members:    oc get ghr -n $NS"
  info "Timeline:       oc get events -n $NS --field-selector reason=Discovery"
  info "The reference model downloads and starts on the 3 GPU nodes in about 5-10 minutes,"
  info "then learning runs (~200 checks per GPU pairing) before detection starts."
fi
