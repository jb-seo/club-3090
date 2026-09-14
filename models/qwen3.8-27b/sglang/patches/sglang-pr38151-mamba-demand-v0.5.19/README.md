# SGLang Mamba eviction demand (v0.5.19)

Packages alphabetc1's commit `ad60252754e895f857b9b8015570245f3dfc2b0a`.
Upstream status and retirement condition: [SGLang #38151 in the tracker](../../../../../../docs/UPSTREAM.md#sglang-sgl-projectsglang).

The patch contains the upstream `allocation.py` change without modification.
It reserves only the Mamba slots that a prefill batch will allocate, preserving
cached checkpoints that the former per-request multiplier evicted unnecessarily.
With overlapping extra buffers, new / prefix-matched / continuing chunked
requests need 3 / 2 / 0 slots respectively. Lazy and no-buffer modes use their
actual allocation sizes too. The upstream CPU regression tests remain in the
SGLang PR; this container patch installs only the runtime file.

All Qwen3.8-27B SGLang composes (dual, multi4, multi8; AutoRound and FP8;
MTP and DFlash2) mount this directory read-only at `/etc/club3090/mamba-demand`
and run `bash /etc/club3090/mamba-demand/install.sh` before launching the server.
This fix is always enabled, independently of the optional W4A8 patch.
The image remains pinned to `lmsysorg/sglang:v0.5.19`.

The installer checks every patch hunk in reverse to detect an already-patched
container, so restarts are idempotent. It preflights forward application before
writing and refuses partial or incompatible source states. `git apply` works
with or without `.git`; `patch(1)` is the fallback. No download is needed.

To inspect an existing container without changing it:

```bash
docker exec <container> bash /etc/club3090/mamba-demand/install.sh --verify
```

For a local source tree, set `SGLANG_DIR=/path/to/sglang` when invoking the
installer. Recreate the container after updating compose so the new mount and
startup command take effect (`docker compose -f <compose.yml> up -d --force-recreate`).

Validation scope: patch application and repeat installation on v0.5.19 source,
drift rejection, and compose configuration checks. Live model serving and
performance have not been validated for this packaging change.
