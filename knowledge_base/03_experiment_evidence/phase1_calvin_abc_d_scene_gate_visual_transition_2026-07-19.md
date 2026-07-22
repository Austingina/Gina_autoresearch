# CALVIN ABC-to-D Scene Gate and Visual Transition

Date: 2026-07-19
Status: visual seed-0 scene gate complete; source-scene health diagnostic next

## Question

Does compact oracle `G=[A(3),F(3)]` retain causal control value when source
demonstrations come from official CALVIN environments A/B/C and evaluation
uses environment D? What policy interface should be carried into the frozen
pretrained-vision stage?

## Data

- Official `task_ABC_D.zip`: 518 GB, SHA256 verified against the CALVIN release
  checksum `c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74`.
- Compact source subset: 2,101 segments and 51,594 frames from
  `push_red_block_left`, `push_blue_block_right`, and both pink-block
  directions.
- Held-out combinations in environment D: `push_red_block_right` and
  `push_blue_block_left`.
- Every object color and direction value appears in training, but the two
  evaluated color-direction combinations do not.
- All closed-loop rows use 100 identical CALVIN initial states per held-out
  task, H16 training, H1 replanning, 160 simulator steps, and seed 0.

## Language Audit

The first implementation used task one-hot vectors. This is not a credible
semantic language baseline because held-out task dimensions never activate in
training. The executor now supports official 384-D CALVIN MiniLM embeddings.
Training and D evaluation load embeddings from their respective official
annotations.

MiniLM condition normalization has moderate shift rather than catastrophic
numeric explosion. Source task z-score norms are 10.5--16.9; held-out push
norms are 22.0--22.2 with maximum absolute coordinate 3.6. This can contribute
to OOD error but does not explain the architecture-level result by itself.

## Closed-Loop Results

| Executor input | Blue left | Red right | Total | Interpretation |
| --- | ---: | ---: | ---: | --- |
| full state + MiniLM, no G | 3/100 | 41/100 | 44/200 | strongest direct-language row |
| full state + MiniLM + G | 0/100 | 0/100 | 0/200 | direct fusion collapses on held-out combinations |
| proprio + MiniLM, no G | 0/100 | 0/100 | 0/200 | object localization is unavailable |
| proprio + MiniLM + G | 0/100 | 0/100 | 0/200 | direct language still interferes with G composition |
| full state + G, no direct language | **20/100** | **15/100** | **35/200** | balanced, causal scene transfer |
| full state + zero G | 3/100 | 2/100 | 5/200 | most normal-G successes disappear |
| full state + sign-flipped G | 0/100 | 0/100 | 0/200 | direction semantics are necessary |
| proprio + G, no direct language | 0/100 | **47/100** | 47/200 | G drives red-right but blue-left remains unsupported |
| proprio + zero/sign-flipped G | 0/100 | 0/100 | 0/200 | red-right behavior is G-causal |

The earlier one-hot/proprio row reached 100/100 on red-right and 0/100 on
blue-left, with zero/flip controls at zero. This confirms that a task encoding
choice can move the executor into very different control basins and must not
be treated as a neutral implementation detail.

## Conclusion

The official scene gate supports a narrower but defensible claim:

```text
oracle G carries reusable, causally effective control information across
CALVIN scenes, but direct language+G fusion does not yet dominate a strong
language-only baseline on every held-out combination.
```

The failure is not evidence that language is unnecessary. The intended causal
architecture is factorized:

```text
language + visual observation -> predicted G
visual/proprio history + G     -> low-level action policy
```

Feeding language directly to the executor duplicates direction and object
information already represented by oracle G. With only four source sentence
conditions, the decoder can memorize complete language/G combinations rather
than learn the reusable interface. The visual stage therefore carries forward
both `visual+oracle-G` and matched `visual+MiniLM/no-G`; direct
`visual+MiniLM+G` is retained as a fusion-ablation failure, not the default
architecture.

## Pretrained Visual Stage

Frozen `facebook/dinov2-small` now encodes CALVIN static and gripper cameras
with shared weights. Each camera contributes a 384-D CLS feature. Training
features use offline caching; evaluation uses the same frozen encoder and
ImageNet normalization online.

- 128-frame cache and online rollout smoke tests passed.
- Full cache completed for all 51,594 source frames as float16 shards.
- Formal seed-0 training completed for:
  - visual + oracle G, no direct executor language;
  - visual + MiniLM, G dropped on every training sample.
- Final offline validation metrics are:
  - visual + oracle G: chunk MSE `0.04217`, first-action MSE `0.02312`;
  - visual + MiniLM/no-G: chunk MSE `0.05548`, first-action MSE `0.02734`.
- The first online evaluation attempt exposed a server-rendering failure, not
  a policy failure. CALVIN initialized an unused tactile camera whose PyBullet
  background render aborted inside `gripper_camera.py:getCameraImage` after an
  environment reset. Visual evaluation now constructs the official environment
  with only the consumed static and gripper cameras; a two-reset smoke test
  completes without the abort.
- Renderer profiling separates the remaining costs: frozen DINOv2 plus policy
  inference takes `0.01--0.09 s/step`. Combining a remapped
  `CUDA_VISIBLE_DEVICES=6` namespace with physical EGL id 6 caused black frames
  and about `18.1 s/env-step`. EGL device 16 (`llvmpipe`) returned complete
  frames at about `0.053--0.056 s/env-step` and was used for the completed D
  gate. The corrected NVIDIA protocol leaves every CUDA device visible and
  uses the same physical id for policy (`cuda:N`) and EGL. It returns complete
  frames; NVIDIA-vs-llvmpipe pixel MAE on the same D state is only `1.32`
  static and `0.76` gripper.
- All four fresh D evaluations completed on the validated EGL-16 path, using
  the same 100 official sequence indices per held-out task:

| Visual executor | Blue left | Red right | Total |
| --- | ---: | ---: | ---: |
| oracle G | 1/100 | 11/100 | **12/200** |
| zero G | 0/100 | 0/100 | 0/200 |
| sign-flipped G | 0/100 | 0/100 | 0/200 |
| MiniLM, no G | 3/100 | 0/100 | 3/200 |

Normal G has 12 normal-only and zero control-only successes against both zero
and sign-flipped G (paired two-sided exact McNemar `p=4.88e-4`). Against the
matched language-only executor it has 12 G-only, 3 language-only, and zero
shared successes (`p=0.0352`). Oracle G therefore remains causally effective
after replacing privileged scene state with frozen visual features, but the
absolute success rate and color-direction asymmetry fail a robust scene gate.
All four rows used the same llvmpipe backend, so normal/zero/sign-flip causal
comparisons remain matched. Absolute policy quality should still be rechecked
with corrected NVIDIA EGL to remove train/render backend shift.
The next diagnostic must measure source-scene closed-loop health before
changing G: low source health would identify the visual executor, whereas high
source health with low D performance would identify scene transfer.

Source-scene reconstruction uses official `scene_info.npy`, per-scene CALVIN
YAML, and each demonstration's first robot/scene state. Scene A reset fidelity
is effectively exact under corrected NVIDIA EGL (median DINO cosine `0.99981`
static and `0.99961` gripper). B/C do not yet reproduce their stored frames
closely enough and are excluded pending configuration recovery. On the same 25
starts per source task, visual+G reaches `72/100` and visual+MiniLM/no-G reaches
`62/100`. Per-task G/language results are blue-right `21/15`, pink-left
`15/18`, pink-right `18/17`, and red-left `18/12`. Paired outcomes are 22
G-only, 12 language-only, 50 both, and 16 neither (`p=0.121`), so both visual
executors are healthy on exact source support but G does not significantly
beat language there. A corrected-NVIDIA D rerun is active in separate
`*_nvidia_egl` result directories.

Server artifacts:

```text
/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D_compact_blocks
/mnt/nvme33n1/wuk_code/calvin_g/data/task_ABC_D_compact_blocks_dinov2_small
/mnt/nvme33n1/wuk_code/AutoResearch/logs/M1_abc_d_scene_gate_none
/mnt/nvme33n1/wuk_code/AutoResearch/logs/M1_abc_d_scene_gate_minilm
/mnt/nvme33n1/wuk_code/AutoResearch/logs/M2_abc_d_visual_dinov2
```

## Next Gate

1. Finish the corrected-NVIDIA D gate for normal/zero/sign-flipped G and the
   matched language-only executor.
2. Replace global DINOv2 CLS vectors with spatial patch tokens and a temporal
   policy only if source-scene health is also weak; do not redesign compact G
   from this result alone.
3. Repeat the D scene gate after the executor passes source health, then train
   language+video to G and run the same interventions.
