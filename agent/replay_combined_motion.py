#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of a combined digit motion dataset on a robot.

This script is consistent with lerobot-replay but designed for locally generated
combined digit motion data.

Examples:

```shell
python replay_combined_motion.py \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=digit_writer \
    --dataset.repo_id=local \
    --dataset.root=./shubhamt0802/combined_number_18 \
    --dataset.episode=0
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. For local datasets, use 'local' or any identifier.
    repo_id: str = "local"
    # Episode to replay.
    episode: int = 0
    # Root directory where the dataset is stored (e.g. './shubhamt0802/combined_number_18').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    # Filter dataset to only include frames from the specified episode since episodes are chunked in dataset V3.0
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.episode)
    actions = episode_frames.select_columns(ACTION)

    # Use dataset fps if not overridden
    fps = cfg.dataset.fps if cfg.dataset.fps is not None else dataset.fps

    robot.connect()

    # Get number info from dataset if available
    number_info = ""
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'info'):
        info = dataset.meta.info
        if "number_produced" in info:
            number_info = f" for number {info['number_produced']}"

    log_say(f"Replaying episode{number_info}", cfg.play_sounds, blocking=True)
    
    for idx in range(len(episode_frames)):
        start_episode_t = time.perf_counter()

        action_array = actions[idx][ACTION]
        action = {}
        for i, name in enumerate(dataset.features[ACTION]["names"]):
            action[name] = action_array[i]

        robot_obs = robot.get_observation()

        processed_action = robot_action_processor((action, robot_obs))

        _ = robot.send_action(processed_action)

        dt_s = time.perf_counter() - start_episode_t
        precise_sleep(1 / fps - dt_s)

    robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
