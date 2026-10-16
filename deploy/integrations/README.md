# Vendor integrations

Dual-Key does not replace the GPU vendors' health tooling. It adds one signal those tools
cannot produce: *this GPU's answers disagree with every peer running the same model*.
It hands that signal to them through a standard Kubernetes mechanism, a **node condition**:

```
DualKeyInferenceIntegrity=True   reason: SilentDataCorruptionSuspected
```

| Vendor | Integration | What happens |
|---|---|---|
| NVIDIA | `nvsentinel-values.yaml` (Kubernetes Object Monitor policy; installed by `scripts/install-nvsentinel.sh`) | NVSentinel raises a fatal health event; its quarantine, drain and remediation run as for any DCGM/XID fault |
| AMD | `amd-remediation-mapping.yaml` (auto-remediation mapping) | AMD GPU Operator runs its workflow: taint, drain, reboot, RVS validation, untaint |
| Any | `GPUIntegrityIncident` objects | Evidence, vendor telemetry, vendor diagnostic result and every action, for ticketing or audit |

Set `spec.remediation.action: handoff` on the DualKeyCluster to let the vendor tooling do the
node work. Dual-Key still:

1. opens a `GPUIntegrityIncident` with the evidence,
2. snapshots the vendor telemetry for that node (DCGM / AMD metrics exporter, node conditions),
3. runs the vendor's own diagnostic on that node (`dcgmi diag -r 1` / `amd-smi`),
4. releases the incident when the node reboots, and
5. re-checks the GPU against its peers before resolving it.

Other policies (`alert`, `cordon`, `drain`, `evict`) let Dual-Key act on its own when no vendor
remediation is installed.
