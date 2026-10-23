# Operation 20261020-POST-REBOOT-CALVIN-G-VALIDATION

- **Operation ID:** `20261020-post-reboot-calvin-g-validation`
- **Status:** `READY_FOR_REBOOT`
- **Created:** `2026-10-20T17:17:24+08:00`
- **Scope:** reboot to load matching NVIDIA kernel module, then validate CALVIN EGL and corrected online G before any formal evaluation.

## Pre-reboot evidence

- NVIDIA userspace/NVML: `595.99`; loaded kernel module: `595.91.07`; `nvidia-smi` exits 18.
- LIBERO 10-epoch training completed at `2026-10-20 16:20:19+08:00`; no LIBERO trainer is active.
- CALVIN original-data downloader was explicitly terminated before reboot; no extractor PID remained after a 10-second check.
- Latest resumable CALVIN state before the final pause is preserved under `data/calvin/task_D_D_original_full_state/` and the evidence directory; restart validates existing file size/CRC and downloads only missing members.
- RoboTwin source and XPolicyLab submodule are cloned; the isolated Python 3.10 environment exists at `/home/ai-sz-a26394-u1/RoboTwin_runtime/env`.

## Ordered post-reboot procedure

1. Verify the loaded NVIDIA kernel module and userspace library match; require `nvidia-smi` exit 0.
2. Verify PyTorch CUDA in the existing CALVIN environment with a bounded matrix operation.
3. Run the smallest CALVIN state-only environment smoke: create environment, reset once, execute one no-motion/zero-action step, close, and preserve exit/log evidence.
4. Run a corrected-G trace on a small fixed-reset block-push set using the object-surface/volume builder; log initial object center, per-step flow progress, transition frame, attraction before/after, `contact_switch_count`, and success.
5. Require at least one valid post-contact transition and confirm post-transition `A=0` while `F` remains the intended object-flow direction; otherwise stop and diagnose rather than launch formal evaluation.
6. Evaluate the existing expanded-data checkpoint on identical tasks and reset states with old `generated_query` and corrected online G; compare approach, contact, switch, and success counts.
7. Only after the matched smoke passes, run the planned same-task, fixed-reset, multi-seed formal evaluation and report task/seed means, worst case, variance, and normal/zero/sign-flip controls.
8. Continue RoboTwin dependency/asset installation and bounded simulator verification after the CALVIN gate.
9. Resume the original CALVIN download only when desired by rerunning the recorded extractor command; verify resumed growth and final CRC/manifest.

## Safety gates

- Do not resume the downloader automatically during reboot validation.
- Do not start formal evaluation if `nvidia-smi`, CUDA, EGL, reset/step, or corrected-G transition evidence fails.
- Do not report the old 13.5%/15.6% expanded-data results as corrected-G results.
- Preserve source/held-out boundaries and use identical task/reset pools for old-versus-corrected comparisons.

## Completion evidence required

- driver versions and `nvidia-smi` output;
- CUDA smoke log and exit code;
- EGL/CALVIN create-reset-step-close log and exit code;
- corrected-G trace with nonzero switch count;
- matched old/new summary and exact artifact paths;
- final operation status (`PASS`, `BLOCKED`, or `FAILED`).

## Actual results

`PENDING POST-REBOOT`
