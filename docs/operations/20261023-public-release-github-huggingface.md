# 2026-10-23 GitHub + Hugging Face public release

## Release decision

- Selected plan: public GitHub code repository plus public Hugging Face derived-data repository.
- Official CALVIN, LIBERO, RoboTwin, ManiSkill, and VIMA payload mirrors (about 90 GB locally) were intentionally excluded.
- Published data consists of canonical semantic indexes, intent/phase labels, exact pinned-source acquisition/selection/checksum manifests, and the five genuinely new RoboTwin online-native expert HDF5 trajectories.

## Remote targets

- GitHub: <https://github.com/Austingina/Gina_autoresearch>
- GitHub code release commit: `3fe4db288a63ddef3e08818884a30fdc2d7e17d9`
- Hugging Face: <https://huggingface.co/datasets/AUSTINGINA/Gina_autoresearch>
- Hugging Face final dataset revision: `4eb16999cb1984bdfa28cf13117413477a9fd783`
- Release verification time: `2026-10-23T18:15:27+08:00`

The dataset contains `CODE_REVISION.json`, which pins it to the GitHub code release commit above. A later GitHub audit-only commit adds this operation record but does not change the released data-processing code.

## Exclusions and safety checks

Excluded from GitHub and the HF derived release:

- checkpoints and model weights (`*.pt`, `*.pth`, `*.ckpt`);
- caches and local virtual environments;
- runtime overlays, logs, videos, temporary files, and experiment products;
- the approximately 90 GB local mirrors of official upstream datasets;
- credentials, tokens, cookies, and private keys.

Pre-publish checks:

- staged secret-pattern matches: `0`;
- HF staging secret-pattern matches: `0`;
- forbidden staged paths: `0`;
- staged files larger than 10 MiB: `0`;
- changed/new Python files compiled: `95`, failures: `0`.

## Tests

- Data construction/index tests: `78/78 PASS` in `.venv-calvin-unit` after installing test-only dependencies into the ignored local environment.
- Strategy scripts passing directly in the CALVIN/PyTorch runtime: matched-task plan, same-G plan, manifest-exact selection, NumPy pickle compatibility, and state-only CALVIN/EGL environment.
- `test_profiled_eval_no_recursion.py` was not claimed as passing: it was blocked during import because the existing CALVIN environment had `huggingface-hub==1.25.1`, while its installed `transformers` required `<1.0`. This dependency mismatch was not changed as part of the release.

## Hugging Face contents and verification

Final remote inventory:

- Hub file count: `39` = 37 manifest-managed payload/metadata files + `MANIFEST_SHA256.csv` + Hub-generated `.gitattributes`;
- manifest-managed bytes: `327,305,815`;
- Hub reported storage: `326,979,336` bytes;
- canonical official inventory: 26,193 successful parents, 117 simulator-qualified tasks, and 2,507,756 native transitions/frames;
- online-native payload: five RoboTwin `place_empty_cup` trajectories.

Verification performed against the fixed HF revision:

1. Downloaded the entire dataset into an independent local directory.
2. Recomputed SHA-256 and byte size for every one of the 37 manifest rows.
3. Result: `0` missing files and `0` size/hash mismatches.
4. Tested unauthenticated public downloads:
   - HF README: HTTP 200;
   - HF manifest: HTTP 200;
   - first 64 bytes of a 9,016,040-byte HDF5: HTTP 206 with the expected content range;
   - GitHub README at the pinned code commit: HTTP 200.

## Reproducibility boundary

The source manifests retain upstream repository/URL, fixed revision, native path/member name, task or trajectory selection, size, and SHA-256/CRC where provided. They are sufficient to reconstruct the official payload selection while the pinned upstream revisions remain available. The multi-source publication is a semantic inventory and provenance release; it does not merge simulator-native state/action vectors and does not claim that all simulators have completed six-dimensional-`G` closed-loop evaluation.
