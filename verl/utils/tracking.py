# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
A unified tracking interface that supports logging data to different backend
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Union

from .logger.aggregate_logger import LocalLogger


class Tracking:
    supported_backend = ["wandb", "mlflow", "swanlab", "console", "tensorboard"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]

        for backend in default_backend:
            assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "wandb" in default_backend:
            import wandb  # type: ignore

            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import mlflow  # type: ignore

            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(config)
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab  # type: ignore

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "console" in default_backend:
            self.console_logger = LocalLogger()
            self.logger["console"] = self.console_logger
        
        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)


    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)

        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics=data, step=step)

class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_root = os.environ.get("TENSORBOARD_ROOT", "tensorboard_log")
        tensorboard_dir = os.path.join(tensorboard_root, project_name, experiment_name)
        self.tensorboard_dir=tensorboard_dir
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        print(f"[TB LOG] dir={self.tensorboard_dir}, step={step}, n_keys={len(data)}")
        for key in data:
            self.writer.add_scalar(key, data[key], step)
        self.writer.flush()

    def finish(self):
        self.writer.close()


@dataclass
class ValGenerationsLogger:
    def log(self, loggers: List[str], samples: List[Tuple[str, str, float]], step: int):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

    def log_generations_to_wandb(self, samples: List[Tuple[str, str, float]], step: int) -> None:
        """Log samples to wandb as a table"""
        import wandb  # type: ignore

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples: List[Tuple[str, str, float]], step: int) -> None:
        """Log samples to swanlab as text"""
        import swanlab  # type: ignore

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"input: {sample[0]}\n\n---\n\noutput: {sample[1]}\n\n---\n\nscore: {sample[2]}"
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)
    
    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_root = os.environ.get("TENSORBOARD_ROOT", "tensorboard_log")
            tensorboard_dir = os.path.join(tensorboard_root, "validation")
            os.makedirs(tensorboard_dir, exist_ok=True)
            print(f"Saving tensorboard log to {tensorboard_dir}.")
            self.writer = SummaryWriter(tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
