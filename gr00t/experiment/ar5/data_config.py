"""GR00T N1.5 data config for synchronized dual-AR5 B camera episodes.

The locked N1.5 ``load_data_config`` accepts a class with ``modality_config``
and ``transform`` methods. Keep the modality declaration light: importing the
N1.5 training transform also constructs its Eagle processor, which is not
needed to inspect a dataset or decode one batch.

Use as an external config from the XRoboToolkit repository root::

    scripts.misc.ar5_b_data_config:AR5BDataConfig

State is 18 measured hinge positions: seven joints per arm and both actual
finger hinges per gripper. Action is 16 position-actuator targets: seven per
arm and one commanded close angle per gripper. All angles are in radians.
"""

from __future__ import annotations


class AR5BDataConfig:
    """Follow the locked GR00T N1.5 ``BaseDataConfig`` interface."""

    video_keys = [
        'video.cam_left_wrist',
        'video.cam_right_wrist',
        'video.cam_high',
    ]
    state_keys = [
        'state.left_arm',
        'state.right_arm',
        'state.left_hand',
        'state.right_hand',
    ]
    action_keys = [
        'action.left_arm',
        'action.right_arm',
        'action.left_hand',
        'action.right_hand',
    ]
    language_keys = ['annotation.human.action.task_description']
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        return {
            'video': ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            'state': ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            'action': ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            'language': ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        """Use the locked N1.5 G1 transform with AR5 camera/key selection.

        This is intentionally called only by the N1.5 training/inference
        pipeline. The raw loader check calls ``modality_config`` alone.
        """
        from gr00t.experiment.data_config import UnitreeG1DataConfig

        template = UnitreeG1DataConfig()
        template.video_keys = self.video_keys
        template.state_keys = self.state_keys
        template.action_keys = self.action_keys
        template.language_keys = self.language_keys
        template.observation_indices = self.observation_indices
        template.action_indices = self.action_indices
        return template.transform()


class AR5BGoalDataConfig(AR5BDataConfig):
    """B-scene 14+2 absolute pre-compensation targets, all in radians.

    The dataset still observes 18 measured hinge angles. The 16 action
    channels are accepted left/right arm goals and left/right gripper goals;
    the independent executor applies servo and contact compensation once.
    Mode, action expiry, and faults remain executor state, not model outputs.
    This class deliberately leaves the older final-ctrl dataset contract
    available as AR5BDataConfig.
    """

    action_semantics = '14 absolute arm joint goals + 2 absolute gripper goals, rad, before compensation'
