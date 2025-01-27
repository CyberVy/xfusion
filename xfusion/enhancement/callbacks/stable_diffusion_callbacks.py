from typing import Dict, Any
from diffusers.callbacks import PipelineCallback


class GradualCFG(PipelineCallback):

    tensor_inputs = [
        "prompt_embeds",
        "add_text_embeds",
        "add_time_ids",]
    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index
        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )
        if step_index == cutoff_step:
            
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            add_text_embeds = callback_kwargs[self.tensor_inputs[1]]
            add_text_embeds = add_text_embeds[-1:]  # "-1" denotes the embeddings for conditional pooled text tokens

            add_time_ids = callback_kwargs[self.tensor_inputs[2]]
            add_time_ids = add_time_ids[-1:]  # "-1" denotes the embeddings for conditional added time vector

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            callback_kwargs[self.tensor_inputs[1]] = add_text_embeds
            callback_kwargs[self.tensor_inputs[2]] = add_time_ids

        return callback_kwargs
