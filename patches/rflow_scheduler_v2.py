# Copyright 2025 DecoVAE Authors
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
#
# This file is adapted from the MONAI project
# (https://github.com/Project-MONAI/MONAI), licensed under the Apache
# License 2.0. It subclasses `RFlowScheduler` and overrides `step` to add
# optional stochastic noise injection.
"""
RFlowScheduler with optional stochastic noise injection at each step.

Drop-in replacement for monai.networks.schedulers.rectified_flow.RFlowScheduler.
Use by setting `_target_` in config JSON to
`patches.rflow_scheduler_v2.RFlowSchedulerV2`.
"""
from __future__ import annotations

from typing import Union

import torch

from monai.networks.schedulers.rectified_flow import RFlowScheduler


class RFlowSchedulerV2(RFlowScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        next_timestep: Union[int, None] = None,
        stochastic_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "num_inference_steps") or not isinstance(self.num_inference_steps, int):
            raise AttributeError(
                "num_inference_steps is missing or not an integer in the class. "
                "Please run self.set_timesteps(num_inference_steps, device, input_img_size_numel) to set it."
            )

        v_pred = model_output

        if next_timestep is not None:
            next_timestep = int(next_timestep)
            dt: float = float(timestep - next_timestep) / self.num_train_timesteps
        else:
            dt = 1.0 / float(self.num_inference_steps) if self.num_inference_steps > 0 else 0.0

        if stochastic_scale is not None and dt > 0:
            sigma = stochastic_scale * (dt ** 0.5)
            eps = torch.randn_like(sample)
            pred_post_sample = sample + v_pred * dt + sigma * eps
        else:
            pred_post_sample = sample + v_pred * dt

        pred_original_sample = sample + v_pred * timestep / self.num_train_timesteps
        return pred_post_sample, pred_original_sample
