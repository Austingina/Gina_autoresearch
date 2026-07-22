# CALVIN Dataset Summary

- dataset_root: `/mnt/nvme33n1/wuk_code/calvin_g/data/task_D_D`
- max_frames_for_stats: 1000

## training

- scene: calvin_scene_D (hydra)
- scene scalars: ['base__slide', 'base__drawer', 'base__button', 'base__switch', 'lightbulb', 'led']
- object order: ['block_red', 'block_blue', 'block_pink']
- episode files: 512077
- annotations: 5124
- annotation frames: 308918
- families: {'slider': 303, 'light': 602, 'drawer': 303, 'place': 303, 'lift': 1360, 'push_into': 150, 'rotate': 901, 'stack': 302, 'push': 900}
- object colors: {'none': 1963, 'blue': 1056, 'red': 1052, 'pink': 1053}

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
- episode files: 99022
- annotations: 1011
- annotation frames: 60575
- families: {'light': 123, 'lift': 268, 'slider': 61, 'rotate': 178, 'drawer': 59, 'push': 175, 'place': 59, 'stack': 58, 'push_into': 30}
- object colors: {'none': 390, 'pink': 207, 'red': 209, 'blue': 205}

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
