# Lab guide

A walkthrough of the lab used to validate Dual-Key: three NVIDIA GPUs on OpenShift on AWS, a silent
fault injected into one of them, and the finding handed to NVIDIA NVSentinel.

## 1. Cluster and GPUs

- OpenShift 4.x on AWS (validated on 4.22, Kubernetes 1.35), `oc` logged in as cluster-admin.
- Three GPU worker nodes of different types. `scripts/setup-aws-gpus.sh` adds them as MachineSets
  (by default T4 `g4dn.xlarge`, A10G `g5.xlarge`, L4 `g6.xlarge`) and installs Node Feature Discovery and the
  NVIDIA GPU Operator. If an instance type has no capacity or quota, set `GPU_TYPES` (for example
  `GPU_TYPES="g4dn.xlarge g5.2xlarge g6.xlarge"`) and re-run. Any three NVIDIA GPU nodes work; the
  validation run used a T4 (`g4dn.xlarge`), an A10G (`g5.2xlarge`) and an L4 (`g6.8xlarge`).
- The T4 has no bfloat16 support, so the AWS sample runs every model server in float16.

## 2. Deploy Dual-Key

```bash
oc apply -k deploy/
oc start-build dualkey -n dualkey --from-dir=. --follow
oc rollout restart deploy/dualkey-operator -n dualkey
oc apply -f deploy/samples/dualkeycluster-aws.yaml
```

The operator finds no model servers, deploys the reference model on the three GPU nodes, forms one
group, learns the normal gap (about 2 minutes), then starts detecting. Open the dashboard:

```bash
echo https://$(oc get route dualkey-dashboard -n dualkey -o jsonpath='{.spec.host}')
```

For the vendor panel, let OpenShift monitoring collect the DCGM exporter metrics:

```bash
oc label ns nvidia-gpu-operator openshift.io/cluster-monitoring=true --overwrite
oc patch clusterpolicy gpu-cluster-policy --type merge -p '{"spec":{"dcgmExporter":{"serviceMonitor":{"enabled":true}}}}'
```

## 3. Inject a silent fault

`scripts/fault.sh` adds an init container to one model server that flips bits in its copy of the
weights before the model loads (one middle-layer tensor). The server runs normally; its numbers drift.

```bash
bash scripts/fault.sh list                      # GPUs you can target
BIT=9 COUNT=1000000 bash scripts/fault.sh       # T4 by default; each hit multiplies or divides a weight by 16
bash scripts/fault.sh watch                     # states and scores every 15 s
bash scripts/fault.sh undo                      # clean model, release the incident, re-check
```

| BIT | Effect of each flip (bfloat16 exponent) |
|---|---|
| 7 | weight ×2 or ÷2 |
| 8 | ×4 or ÷4 |
| 9 | ×16 or ÷16 |

The T4 restarts (3–6 minutes: model download, bit flips, load), then is usually caught within 2–10
minutes.

## 4. Hand findings to NVIDIA NVSentinel

```bash
brew install helm        # if needed
bash scripts/install-nvsentinel.sh
```

The script checks the cluster, makes sure DCGM runs as a service, uses or installs cert-manager,
grants the privileged SCC to NVSentinel's service accounts (lab setting), creates MongoDB credentials,
installs NVSentinel with fault quarantine, node drainer and the Kubernetes Object Monitor policy, applies
the OpenShift SELinux setting its node agents need (`spc_t`), and switches Dual-Key to `handoff`.

Then repeat step 3 and check:

```bash
bash scripts/install-nvsentinel.sh status   # T4 node: UNSCHEDULABLE=true, DUALKEY=True, NVSentinel state
oc get gii -n dualkey                       # the incident, its policy and vendor diagnostic result
```

## 5. Troubleshooting notes from the lab

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard pod `PermissionError` on `ui.py` | Files copied with owner-only permissions; container runs as a random UID | `chmod -R a+rX .` before building (the Containerfile also fixes permissions) |
| Operator 403 "cannot set blockOwnerDeletion" | OpenShift requires `update` on `<resource>/finalizers` | Included in `deploy/rbac.yaml` |
| Helm: `docker-credential-desktop` not found | Docker config points at Docker Desktop's credential helper | The install script uses empty registry settings for the chart pull |
| NVSentinel `mongodb-0` `CreateContainerConfigError` | Secret `mongodb` missing the replica-set key | The install script adds all four keys |
| NVSentinel platform connectors: `bind: permission denied` on `/var/run/nvsentinel.sock` | SELinux blocks ordinary containers from creating the host socket | The install script sets `seLinuxOptions.type: spc_t` on the socket users |
| Small faults are seen but not flagged | Spread floor (`stdFloor` 0.01) far above the real spread | `oc patch dkc fleet -n dualkey --type merge -p '{"spec":{"detector":{"stdFloor":0.001}}}'` |

## 6. Evidence to keep

```bash
mkdir -p evidence
oc get gii -n dualkey -o yaml > evidence/incident.yaml
oc get ghr -n dualkey -o wide > evidence/gpu-records.txt
bash scripts/install-nvsentinel.sh status > evidence/nvsentinel-status.txt
oc logs -n nvsentinel deploy/fault-quarantine --tail=50 > evidence/nvsentinel-fault-quarantine.txt
```
