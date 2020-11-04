# Copyright 2019 The ROBEL Authors.
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

"""Walk tasks with DKitty robots.

This is a single movement from an initial position to a target position.
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.walk import BaseDKittyWalk
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH_UP = 'robel/dkitty/assets/dkitty_walk_tilted_up-v0.xml'
DKITTY_ASSET_PATH_DOWN = 'robel/dkitty/assets/dkitty_walk_tilted_down-v0.xml'

TILT = 0.0873


@configurable(pickleable=True)
class DKittyWalkTiltedFixedUp(BaseDKittyWalk):
    """Walk straight towards a fixed location."""
    
    def __init__(self, asset_path: str = DKITTY_ASSET_PATH_UP, *args, **kwargs):
        super().__init__(asset_path, *args, **kwargs)

    def _reset(self):
        """Resets the environment."""
        target_dist = 2.0
        target_theta = np.pi / 2  # Point towards y-axis
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta) * np.cos(TILT), np.sin(TILT)
        ])
        super()._reset()
        
        
@configurable(pickleable=True)
class DKittyWalkTiltedFixedDown(BaseDKittyWalk):
    """Walk straight towards a fixed location."""
    
    def __init__(self, asset_path: str = DKITTY_ASSET_PATH_DOWN, *args, **kwargs):
        super().__init__(asset_path, *args, **kwargs)

    def _reset(self):
        """Resets the environment."""
        target_dist = 2.0
        target_theta = np.pi / 2  # Point towards y-axis
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta) * np.cos(-TILT), np.sin(-TILT)
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkTiltedRandomUp(BaseDKittyWalk):
    """Walk towards a random location."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH_UP,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(asset_path, *args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + self.np_random.uniform(
            *self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta) * np.cos(TILT), np.sin(target_theta) * np.sin(TILT)
        ])
        super()._reset()
        
        
@configurable(pickleable=True)
class DKittyWalkTiltedRandomDown(BaseDKittyWalk):
    """Walk towards a random location."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH_DOWN,
            *args,
            target_distance_range: Tuple[float, float] = (1.0, 2.0),
            # +/- 60deg
            target_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            target_distance_range: The range in which to sample the target
                distance.
            target_angle_range: The range in which to sample the angle between
                the initial D'Kitty heading and the target.
        """
        super().__init__(asset_path, *args, **kwargs)
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(*self._target_distance_range)
        # Offset the angle by 90deg since D'Kitty looks towards +y-axis.
        target_theta = np.pi / 2 + self.np_random.uniform(
            *self._target_angle_range)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta) * np.cos(-TILT), np.sin(target_theta) * np.sin(-TILT)
        ])
        super()._reset()


@configurable(pickleable=True)
class DKittyWalkTiltedRandomDynamicsUp(DKittyWalkTiltedRandomUp):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()

        
@configurable(pickleable=True)
class DKittyWalkTiltedRandomDynamicsDown(DKittyWalkTiltedRandomDown):
    """Walk straight towards a random location."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2.8, 3.2),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()
        
