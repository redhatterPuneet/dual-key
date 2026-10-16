#!/usr/bin/env bash
# Install NVIDIA NVSentinel on this OpenShift cluster (lab setup) and hand Dual-Key's findings to it.
#
#   bash scripts/install-nvsentinel.sh            install / upgrade, then switch Dual-Key to handoff
#   bash scripts/install-nvsentinel.sh status     show NVSentinel pods and what it did to GPU nodes
#   bash scripts/install-nvsentinel.sh uninstall  remove NVSentinel, switch Dual-Key back to cordon
#
# After install, the loop is:
#   Dual-Key catches a GPU -> sets node condition DualKeyInferenceIntegrity=True
#   -> NVSentinel's Kubernetes Object Monitor raises a fatal GPU health event
#   -> NVSentinel cordons the node and drains its workloads
#   -> after repair, Dual-Key releases the incident and clears the condition
#   -> NVSentinel sees a healthy event and uncordons; Dual-Key re-checks the GPU.
#
# Lab notes: grants the privileged SCC to NVSentinel's service accounts (its GPU and syslog
# monitors read host devices and logs). NVSentinel's README lists Kubernetes 1.34+
# (OpenShift 4.21+); older clusters get a warning (STRICT=1 stops instead).
#
# Options (environment): NVSENTINEL_VERSION=v1.25.0  NVS_NS=nvsentinel  DK_NS=dualkey
set -euo pipefail

NVSENTINEL_VERSION=${NVSENTINEL_VERSION:-v1.25.0}
CERT_MANAGER_VERSION=${CERT_MANAGER_VERSION:-v1.21.1}
NVS_NS=${NVS_NS:-nvsentinel}
DK_NS=${DK_NS:-dualkey}
GPU_NS=${GPU_NS:-nvidia-gpu-operator}
HERE=$(cd "$(dirname "$0")/.." && pwd)
VALUES="$HERE/deploy/integrations/nvsentinel-values.yaml"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

command -v oc >/dev/null || die "oc not found."
command -v helm >/dev/null || die "helm not found. Install it with: brew install helm"
oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."

dk_cluster() { oc get dkc -n "$DK_NS" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true; }

set_policy() {   # $1 = remediation action for Dual-Key
  local c; c=$(dk_cluster)
  [ -n "$c" ] || { warn "No DualKeyCluster in $DK_NS; set spec.remediation.action: $1 yourself."; return; }
  oc patch dkc "$c" -n "$DK_NS" --type merge \
    -p "{\"spec\":{\"remediation\":{\"action\":\"$1\",\"autoReleaseAfterReboot\":false}}}" >/dev/null
  echo "Dual-Key ($c) response policy: $1"
}

status() {
  say "NVSentinel pods"
  oc get pods -n "$NVS_NS" -o wide 2>/dev/null || echo "not installed"
  say "GPU nodes: cordoned? Dual-Key condition?"
  oc get nodes -l nvidia.com/gpu.present=true \
    -o custom-columns='NODE:.metadata.name,UNSCHEDULABLE:.spec.unschedulable,DUALKEY:.status.conditions[?(@.type=="DualKeyInferenceIntegrity")].status,NVSENTINEL-STATE:.metadata.labels.dgxc\.nvidia\.com/nvsentinel-state'
  say "Dual-Key incidents"
  oc get gii -n "$DK_NS" 2>/dev/null || true
}

uninstall() {
  say "Removing NVSentinel"
  helm uninstall nvsentinel -n "$NVS_NS" || true
  set_policy cordon
  echo "cert-manager and the $NVS_NS namespace were left in place (oc delete ns $NVS_NS to remove)."
}

install() {
  say "1/7 Checking the cluster version"
  local ver minor ocp
  ver=$(oc version -o json | python3 -c 'import json,sys; v=json.load(sys.stdin)["serverVersion"]; print(v["major"]+"."+v["minor"].rstrip("+"))')
  ocp=$(oc get clusterversion version -o jsonpath='{.status.desired.version}' 2>/dev/null || echo "?")
  minor=${ver#*.}
  echo "Kubernetes $ver (OpenShift $ocp)"
  if [ "${minor:-0}" -lt 34 ]; then
    warn "NVSentinel's README lists Kubernetes 1.34+ (OpenShift 4.21+). Continuing as a lab install."
    [ "${STRICT:-0}" = "1" ] && die "STRICT=1: stopping."
  fi

  say "2/7 Making sure DCGM runs as a service (NVSentinel's GPU monitor connects to it)"
  local cp dcgm
  cp=$(oc get clusterpolicy -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [ -n "$cp" ] || die "No NVIDIA ClusterPolicy found. Is the NVIDIA GPU Operator installed?"
  dcgm=$(oc get clusterpolicy "$cp" -o jsonpath='{.spec.dcgm.enabled}')
  if [ "$dcgm" != "true" ]; then
    oc patch clusterpolicy "$cp" --type merge -p '{"spec":{"dcgm":{"enabled":true}}}' >/dev/null
    echo "Enabled DCGM in ClusterPolicy $cp; waiting for the service..."
    for _ in $(seq 1 30); do oc get svc nvidia-dcgm -n "$GPU_NS" >/dev/null 2>&1 && break; sleep 10; done
  fi
  oc get svc nvidia-dcgm -n "$GPU_NS" >/dev/null 2>&1 && echo "DCGM service: nvidia-dcgm.$GPU_NS:5555" \
    || warn "Service nvidia-dcgm not found in $GPU_NS; NVSentinel's GPU monitor may not connect (Dual-Key's loop still works)."

  say "3/7 cert-manager (NVSentinel uses it for internal TLS)"
  if oc get crd certificates.cert-manager.io >/dev/null 2>&1; then
    echo "Using the cert-manager already on the cluster."
  else
    helm repo add jetstack https://charts.jetstack.io --force-update >/dev/null
    helm upgrade --install cert-manager jetstack/cert-manager -n cert-manager --create-namespace \
      --version "$CERT_MANAGER_VERSION" --set crds.enabled=true --wait --timeout 10m
  fi

  say "4/7 Namespace $NVS_NS and permissions"
  oc get ns "$NVS_NS" >/dev/null 2>&1 || oc create ns "$NVS_NS" >/dev/null
  oc adm policy add-scc-to-group privileged "system:serviceaccounts:$NVS_NS" >/dev/null
  echo "Granted the privileged SCC to service accounts in $NVS_NS (lab setup)."

  say "5/7 MongoDB credentials for NVSentinel's state store"
  # The bundled MongoDB (replica set) needs all four keys; add any that are missing.
  if ! oc get secret mongodb -n "$NVS_NS" >/dev/null 2>&1; then
    oc create secret generic mongodb -n "$NVS_NS" \
      --from-literal=mongodb-root-password="$(openssl rand -hex 24)" \
      --from-literal=mongodb-replica-set-key="$(openssl rand -hex 24)" \
      --from-literal=mongodb-passwords="" --from-literal=mongodb-metrics-password="" >/dev/null
    oc annotate secret mongodb -n "$NVS_NS" helm.sh/resource-policy=keep --overwrite >/dev/null
    echo "Created secret mongodb."
  else
    for key in mongodb-replica-set-key mongodb-passwords mongodb-metrics-password; do
      if [ -z "$(oc get secret mongodb -n "$NVS_NS" -o jsonpath="{.data.${key//./\\.}}" 2>/dev/null)" ] && \
         ! oc get secret mongodb -n "$NVS_NS" -o json | grep -q "\"$key\""; then
        val=""; [ "$key" = "mongodb-replica-set-key" ] && val="$(openssl rand -hex 24)"
        oc patch secret mongodb -n "$NVS_NS" --type merge -p "{\"stringData\":{\"$key\":\"$val\"}}" >/dev/null
        echo "Added missing key $key to secret mongodb."
      fi
    done
    echo "Secret mongodb is complete."
  fi

  say "6/7 Installing NVSentinel $NVSENTINEL_VERSION (a few minutes)"
  # The chart is public on ghcr.io. Use empty registry settings so a Docker credential helper
  # configured on this machine (e.g. docker-credential-desktop without Docker Desktop) is not used.
  local regdir; regdir=$(mktemp -d)
  echo '{}' > "$regdir/config.json"
  export DOCKER_CONFIG="$regdir"
  helm upgrade --install nvsentinel oci://ghcr.io/nvidia/nvsentinel \
        --version "$NVSENTINEL_VERSION" -n "$NVS_NS" -f "$VALUES" ${EXTRA:-} \
        --registry-config "$regdir/config.json" --timeout 10m

  # OpenShift: SELinux stops ordinary containers from creating or using the Unix socket that
  # NVSentinel's node agents share in /var/run/nvsentinel on each host. Run those components
  # with the SELinux type for trusted system containers (allowed by the privileged SCC above).
  echo "Applying the OpenShift SELinux setting to the socket-sharing components..."
  for r in ds/platform-connectors ds/syslog-health-monitor-regular deploy/kubernetes-object-monitor; do
    oc get "$r" -n "$NVS_NS" >/dev/null 2>&1 && oc patch "$r" -n "$NVS_NS" --type merge \
      -p '{"spec":{"template":{"spec":{"securityContext":{"seLinuxOptions":{"type":"spc_t"}}}}}}' >/dev/null
  done

  echo "Waiting for NVSentinel to become ready (MongoDB first, a few minutes)..."
  local ok=1
  for r in statefulset/mongodb ds/platform-connectors deploy/kubernetes-object-monitor deploy/fault-quarantine deploy/node-drainer; do
    oc rollout status "$r" -n "$NVS_NS" --timeout=15m >/dev/null 2>&1 && echo "  ready: $r" || { echo "  NOT ready: $r"; ok=0; }
  done
  if [ "$ok" != "1" ]; then
    warn "Some NVSentinel parts are not ready. Pods that are not running:"
    oc get pods -n "$NVS_NS" | grep -v -E 'Running|Completed' || true
    echo "Send me: oc get pods -n $NVS_NS ; oc logs -n $NVS_NS ds/platform-connectors --tail=25"
    exit 1
  fi
  oc get pods -n "$NVS_NS"

  say "7/7 Switching Dual-Key to hand its findings to NVSentinel"
  set_policy handoff

  say "Done. Try the full loop:"
  echo "  BIT=9 COUNT=1000000 bash scripts/fault.sh     # Dual-Key catches the T4, NVSentinel cordons its node"
  echo "  bash scripts/install-nvsentinel.sh status     # see the node cordoned by NVSentinel"
  echo "  bash scripts/fault.sh undo                    # release: NVSentinel uncordons, Dual-Key re-checks"
}

case "${1:-install}" in
  install)   install ;;
  status)    status ;;
  uninstall) uninstall ;;
  *) sed -n '2,20p' "$0" ;;
esac
