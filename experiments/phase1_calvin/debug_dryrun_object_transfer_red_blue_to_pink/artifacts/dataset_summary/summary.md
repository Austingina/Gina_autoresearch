# CALVIN Dataset Summary

- dataset_root: `/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset`
- max_frames_for_stats: 1000

## training

- scene: calvin_scene_D (hydra)
- scene scalars: ['base__slide', 'base__drawer', 'base__button', 'base__switch', 'lightbulb', 'led']
- object order: ['block_red', 'block_blue', 'block_pink']
- episode files: 2771
- annotations: 9
- annotation frames: 503
- families: {'light': 3, 'push': 1, 'place': 2, 'lift': 3}
- object colors: {'none': 5, 'pink': 1, 'red': 2, 'blue': 1}

### Keys

- `actions`: shape=[7] dtype=float64
- `rel_actions`: shape=[7] dtype=float64
- `robot_obs`: shape=[15] dtype=float64
- `scene_obs`: shape=[24] dtype=float64
- `rgb_static`: shape=[200, 200, 3] dtype=uint8
- `rgb_gripper`: shape=[84, 84, 3] dtype=uint8
- `rgb_tactile`: shape=[160, 120, 6] dtype=uint8
- `depth_static`: shape=[200, 200] dtype=float32
- `depth_gripper`: shape=[84, 84] dtype=float32
- `depth_tactile`: shape=[160, 120, 2] dtype=float32

## validation

- scene: calvin_scene_D (hydra)
- scene scalars: ['base__slide', 'base__drawer', 'base__button', 'base__switch', 'lightbulb', 'led']
- object order: ['block_red', 'block_blue', 'block_pink']
- episode files: 1675
- annotations: 8
- annotation frames: 472
- families: {'lift': 4, 'light': 2, 'place': 1, 'push': 1}
- object colors: {'red': 1, 'none': 3, 'blue': 3, 'pink': 1}

### Keys

- `actions`: shape=[7] dtype=float64
- `rel_actions`: shape=[7] dtype=float64
- `robot_obs`: shape=[15] dtype=float64
- `scene_obs`: shape=[24] dtype=float64
- `rgb_static`: shape=[200, 200, 3] dtype=uint8
- `rgb_gripper`: shape=[84, 84, 3] dtype=uint8
- `rgb_tactile`: shape=[160, 120, 6] dtype=uint8
- `depth_static`: shape=[200, 200] dtype=float32
- `depth_gripper`: shape=[84, 84] dtype=float32
- `depth_tactile`: shape=[160, 120, 2] dtype=float32
