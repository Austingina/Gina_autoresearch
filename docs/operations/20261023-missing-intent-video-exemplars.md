# OP-20261023-GINA-MISSING-INTENT-VIDEO-EXEMPLARS-01

- **Created:** 2026-10-23T10:46:09+08:00
- **Status:** `PASS`
- **Target:** one representative video export for each currently unrepresented intent family where accepted official visual observations already exist
- **Authoritative root:** `/home/ai-sz-a26394-u1/Gina_autoresearch`
- **Git branch/HEAD:** `master` / `3d4aeba1badaa3d6ce781033e0d9187729dfec54`
- **Pre-existing worktree:** dirty; do not modify or revert unrelated files

## Baseline

Canonical official inventory `official_reorg_20261023_v6` contains 20 intent families. Existing directory `artifacts/videos/classic_new_intents_20261022/` provides nine representatives:

- `pick_place`
- `long_horizon_composite`
- `placement`
- `planar_push`
- `object_manipulation`
- `object_acquisition`
- `handover`
- `multi_object_arrangement`
- `tool_or_contact_operation`

Eleven families currently have no dedicated video exemplar:

- `orientation_manipulation`
- `insertion_assembly`
- `rolling_contact`
- `block_push`
- `constraint_aware_tool_interaction`
- `long_horizon_rearrangement`
- `ordered_imitation`
- `long_horizon_memory_manipulation`
- `switch_activation`
- `articulation`
- `container_transport`

## Scope

Export nine missing families from complete accepted official parents with existing visual payloads:

- VIMA: orientation, constraint-aware tool interaction, long-horizon rearrangement, ordered imitation, long-horizon memory manipulation;
- LIBERO: switch activation and articulation;
- CALVIN: block push and container transport.

`insertion_assembly` and `rolling_contact` currently exist only in the selected ManiSkill state/action payloads. They remain explicit `BLOCKED_NO_OFFICIAL_VISUAL_PAYLOAD` entries; do not fabricate frames or claim a video. A later persisted ManiSkill/SAPIEN replay would be a separate operation and evidence class.

## Output and no-overwrite policy

```text
artifacts/videos/missing_intents_20261023/
```

The directory must not exist before execution. Existing videos are read-only.

## Gates

1. Source parent must be present in canonical v6 and retain official-success provenance.
2. Use one complete parent per family; no cross-parent phase splicing.
3. Preserve native visual-frame order; do not interpolate task states.
4. Encode H.264/yuv420p/faststart with no synthetic audio.
5. Manifest records family, simulator, task, parent ID, source revision/path, success evidence, source and video hashes, native frame count, probe metadata, and `new_physical_trajectory=false`.
6. `ffprobe` every MP4 and decode early/middle/late frames into a contact sheet.
7. Videos are exports of stored official observations, not new trajectories or online simulator validation.

## Abort conditions

- Output path collision;
- parent absent from canonical index;
- missing or inconsistent success evidence;
- malformed or empty visual stream;
- source hash mismatch;
- black/corrupt/unrecognizable early/middle/late frames;
- any attempt to substitute state-only ManiSkill payloads with synthetic visuals.

## PASS criteria

Nine new verified video exemplars plus a manifest, README, and contact sheet; the two state-only ManiSkill families are explicitly listed as blocked rather than counted as completed.

## Actual results

Completed at `2026-10-23T10:49:34+08:00`.

- Exported **9** previously missing intent-family exemplars from complete accepted official parents:
  - VIMA: `orientation_manipulation`, `constraint_aware_tool_interaction`, `long_horizon_rearrangement`, `ordered_imitation`, `long_horizon_memory_manipulation`;
  - CALVIN: `block_push`, `container_transport`;
  - LIBERO: `switch_activation`, `articulation`.
- Existing 9 families plus these 9 exports provide dedicated-video coverage for **18/20** canonical intent families.
- `insertion_assembly` (2,975 parents) and `rolling_contact` (2,373 parents) remain `BLOCKED_NO_OFFICIAL_VISUAL_PAYLOAD`: the selected official ManiSkill payloads are state/action only. No synthetic visual substitute was created.
- All exports are H.264/yuv420p/faststart without audio. Native visual-frame order and complete parent boundaries are preserved.
- Independent verifier rechecked canonical family membership, official parent success, provenance, video/source hashes, frame counts, codec/pixel format, and decoded early/middle/late frame brightness/variance: `PASS` for 9/9 videos.
- Contact-sheet visual inspection found correct orientation, no black/corrupt frames, visible task-state changes, and unobscured workspaces in all nine rows. VIMA videos are intentionally sparse macro-action observation sequences, not continuous simulator recordings.
- New physical trajectories: **0**.
- Online simulator replays: **0**.

Artifacts:

```text
artifacts/videos/missing_intents_20261023/
├── 10_orientation_vima.mp4
├── 11_constraint_tool_vima.mp4
├── 12_long_rearrangement_vima.mp4
├── 13_ordered_imitation_vima.mp4
├── 14_long_memory_vima.mp4
├── 15_block_push_calvin.mp4
├── 16_switch_activation_libero.mp4
├── 17_articulation_libero.mp4
├── 18_container_transport_calvin.mp4
├── contact_sheet.jpg
├── manifest.json
└── README.md
```

Evidence hashes:

- `manifest.json`: `f363bed80f7e171bf3b153b80c2996aac92ee44f5e5ec1a6ac61a142a26a308d`
- `contact_sheet.jpg`: `7bfb63fde9b2a9295b1511d09dd17fac1e5a4b460dd783710618d538dc6ac13e`
- independent readback log: `2cb013f53ade5d2c57e9e4dc147cf1b1256533071c1ca264f60e3677ec1b64ac`

The next allowed operation for the two blocked families is a separately authorized ManiSkill/SAPIEN persisted replay with runtime, render-stack, native success, and artifact-readback gates. It must not be described as an offline official-video export.
