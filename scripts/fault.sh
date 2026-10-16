#!/usr/bin/env bash
# Dual-Key lab: make one GPU "faulty" and watch Dual-Key catch it.
#
# The fault is silent data corruption: bits are flipped in that GPU's copy of the
# model weights before the model loads. The server keeps answering normally, but
# its numbers are slightly off, like a GPU with a bad memory cell.
#
#   bash scripts/fault.sh                 inject into the T4 (or the first GPU found) and watch
#   bash scripts/fault.sh inject [GPU]    GPU = part of a name or GPU type, e.g. T4, L4, 1-116
#   bash scripts/fault.sh watch           show states and scores every 15 seconds
#   bash scripts/fault.sh undo [GPU]      release the GPU, remove the fault, restart clean
#   bash scripts/fault.sh list            GPUs you can target
#
# Options (environment): COUNT=20000 (weights to corrupt), BIT=7, NS=dualkey
set -euo pipefail

NS=${NS:-dualkey}
COUNT=${COUNT:-20000}
BIT=${BIT:-7}
IMAGE="image-registry.openshift-image-registry.svc:5000/${NS}/dualkey:latest"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
die()  { printf '\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

oc whoami >/dev/null 2>&1 || die "Not logged in. Run 'oc login ...' first."

# one line per model server: deployment node product
targets() {
  for d in $(oc get deploy -n "$NS" -o name | grep '/dk-model-' || true); do
    node=$(oc get "$d" -n "$NS" -o jsonpath='{.spec.template.spec.nodeSelector.kubernetes\.io/hostname}')
    prod=$(oc get node "$node" -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.product}{.metadata.labels.amd\.com/gpu\.product}' 2>/dev/null || true)
    echo "${d#deployment.apps/} ${node:-?} ${prod:-unknown}"
  done
}

pick() {   # $1 = optional match; prints "deployment node product"
  local all line=""
  all=$(targets)
  [ -n "$all" ] || die "No Dual-Key model servers (dk-model-*) found in namespace $NS."
  if [ -n "${1:-}" ]; then
    line=$(echo "$all" | grep -i -- "$1" | head -1 || true)
  else
    line=$(echo "$all" | grep -i 't4' | head -1 || true)
    [ -n "$line" ] || line=$(echo "$all" | head -1)
  fi
  echo "$line"
}

member_for() {   # GPUHealthRecord name for a node (Dual-Key's name for that GPU)
  local m
  m=$(oc get ghr -n "$NS" -o jsonpath="{range .items[?(@.spec.node==\"$1\")]}{.metadata.name}{'\\n'}{end}" 2>/dev/null | head -1 || true)
  [ -n "$m" ] || m=$(oc get ghr -n "$NS" -o name 2>/dev/null | sed 's#.*/##' | grep -- "${1%%.*}" | head -1 || true)
  echo "${m:-$2}"
}

init_index() {   # index of the fault-injection init container, or empty
  oc get deploy "$1" -n "$NS" -o jsonpath='{range .spec.template.spec.initContainers[*]}{.name}{"\n"}{end}' \
    | awk '$0=="fault-injection"{print NR-1; exit}'
}

remove_fault() {
  local idx; idx=$(init_index "$1")
  [ -z "$idx" ] && return 1
  oc patch deploy "$1" -n "$NS" --type json -p "[{\"op\":\"remove\",\"path\":\"/spec/template/spec/initContainers/$idx\"}]" >/dev/null
}

dashboard() {
  local h; h=$(oc get route dualkey-dashboard -n "$NS" -o jsonpath='{.spec.host}' 2>/dev/null || true)
  [ -n "$h" ] && echo "Dashboard: https://$h"
}

watch_states() {   # $1 = optional member to stop on when quarantined; $2 = timeout minutes
  local member=${1:-} limit=$(( ${2:-25} * 60 )) start=$SECONDS
  while :; do
    printf '\n%s\n' "$(date +%H:%M:%S)"
    oc get ghr -n "$NS" -o custom-columns='GPU:.metadata.name,STATE:.status.state,SCORE:.status.score,CHECKS:.status.checks' 2>/dev/null || true
    if [ -n "$member" ]; then
      st=$(oc get ghr "$member" -n "$NS" -o jsonpath='{.status.state}' 2>/dev/null || true)
      if [ "$st" = "Quarantined" ]; then
        say "Caught: $member is Quarantined."
        echo "Undo with: bash scripts/fault.sh undo ${HINT:-}"
        return 0
      fi
    fi
    if [ $((SECONDS - start)) -ge "$limit" ]; then
      say "Not caught after $(( limit / 60 )) minutes."
      echo "The fault may be too small to stand out. Try a bigger one:"
      echo "  bash scripts/fault.sh undo ${HINT:-} && COUNT=1000000 bash scripts/fault.sh inject ${HINT:-}"
      return 1
    fi
    sleep 15
  done
}

inject() {
  read -r dep node prod <<<"$(pick "${1:-}")"
  [ -n "${dep:-}" ] || die "No matching model server for '${1:-}'. See: bash scripts/fault.sh list"
  member=$(member_for "$node" "${dep#dk-model-}")
  HINT=${node%%.*}
  say "Injecting a silent fault into $member (${prod}, node ${node%%.*})"
  case "$BIT" in 7) F=2;; 8) F=4;; 9) F=16;; 10) F=256;; *) F="";; esac
  echo "Flipping bit $BIT in $COUNT weights of a middle layer${F:+ (each hit multiplies or divides a weight by $F)}."

  vol=$(oc get deploy "$dep" -n "$NS" -o jsonpath='{.spec.template.spec.initContainers[0].volumeMounts[?(@.mountPath=="/models")].name}')
  [ -n "$vol" ] || die "Could not find the /models volume on $dep."

  remove_fault "$dep" && echo "Replaced an earlier fault on $dep."
  oc patch deploy "$dep" -n "$NS" --type json -p "[
    {\"op\":\"add\",\"path\":\"/spec/template/spec/initContainers/-\",\"value\":{
      \"name\":\"fault-injection\",\"image\":\"$IMAGE\",
      \"command\":[\"python\",\"-m\",\"dualkey.faultinject\",\"--model-dir\",\"/models/model\",
                   \"--bit\",\"$BIT\",\"--count\",\"$COUNT\",\"--seed\",\"7\",
                   \"--ground-truth\",\"/models/ground_truth.json\"],
      \"env\":[{\"name\":\"DK_GPU_NAME\",\"value\":\"$member\"}],
      \"volumeMounts\":[{\"name\":\"$vol\",\"mountPath\":\"/models\"}]}}]" >/dev/null

  say "Step 1 of 2: restarting $member with the corrupted model (re-downloads the model, about 3-6 minutes)"
  dashboard
  oc rollout status deploy/"$dep" -n "$NS" --timeout=15m
  oc logs -n "$NS" deploy/"$dep" -c fault-injection 2>/dev/null | tail -3 || true

  say "Step 2 of 2: waiting for Dual-Key to catch it (usually 2-10 minutes)"
  echo "On the dashboard: $member's line climbs in 'Score drift', its lines turn red in 'Who agrees with whom'."
  watch_states "$member" 25
}

open_incident() {   # newest unresolved GPUIntegrityIncident for a member
  oc get gii -n "$NS" --sort-by=.metadata.creationTimestamp \
    -o custom-columns='N:.metadata.name,G:.spec.gpu,P:.status.phase' --no-headers 2>/dev/null \
    | awk -v m="$1" '$2==m && $3!="Resolved" {n=$1} END {if (n) print n}'
}

undo() {
  read -r dep node prod <<<"$(pick "${1:-}")"
  [ -n "${dep:-}" ] || die "No matching model server for '${1:-}'."
  member=$(member_for "$node" "${dep#dk-model-}")

  say "Step 1 of 2: removing the fault and restarting $member with a clean model"
  if remove_fault "$dep"; then
    oc rollout status deploy/"$dep" -n "$NS" --timeout=15m
  else
    echo "No fault was installed on $dep."
  fi

  inc=$(open_incident "$member")
  if [ -n "$inc" ]; then
    say "Step 2 of 2: releasing incident $inc (taint and node condition removed, then $member is re-checked)"
    oc patch gii "$inc" -n "$NS" --type merge -p '{"spec":{"release":true}}' >/dev/null
    for _ in $(seq 1 60); do
      line=$(oc get gii "$inc" -n "$NS" -o jsonpath='{.status.phase}{"|"}{.status.validation.progress}' 2>/dev/null || true)
      printf '%s  %s\n' "$(date +%H:%M:%S)" "${line/|/  -  }"
      case "${line%%|*}" in
        Resolved) say "Resolved: $member passed re-validation."; break ;;
        ValidationFailed|ValidationTimedOut) say "${line%%|*}: see 'oc describe gii $inc -n $NS'"; break ;;
      esac
      sleep 15
    done
  else
    say "Step 2 of 2: no open incident for $member; clearing its record directly"
    oc patch ghr "$member" -n "$NS" --type merge -p '{"spec":{"release":true}}' >/dev/null 2>&1 || true
    oc adm taint node "$node" dualkey.io/suspect- >/dev/null 2>&1 || true
  fi
  dashboard
}

case "${1:-inject}" in
  inject) shift || true; inject "${1:-}" ;;
  undo)   shift; undo "${1:-}" ;;
  watch)  watch_states "" 600 ;;
  list)   echo "DEPLOYMENT NODE GPU"; targets ;;
  -h|--help|help) sed -n '2,16p' "$0" ;;
  *) inject "$1" ;;
esac
