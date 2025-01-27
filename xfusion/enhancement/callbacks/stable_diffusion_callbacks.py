from typing import Dict, Any
from diffusers.callbacks import PipelineCallback


class GradualCFG(PipelineCallback):

    tensor_inputs = []
    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index
        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )
        if step_index == cutoff_step:
            pipeline._guidance_scale = 0.0

        return callback_kwargs
