import abc
import inspect
import logging
import pickle
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type

import hydra
import numpy as np
import omegaconf
from torch.utils.tensorboard import SummaryWriter


class Logger(AbstractContextManager):
    def __init__(self, save_checkpoint: bool, checkpoint_file_name: str = "training_state"):
        self.save_checkpoint = save_checkpoint
        self.checkpoint_file_name = checkpoint_file_name

    @abc.abstractmethod
    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        """Write a dictionary of metrics to the logger.

        Args:
            data: dictionary of metrics names and their values.
            label: optional label (e.g. 'train' or 'eval').
            env_steps: optional env step count.
        """

    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    def upload_checkpoint(self) -> None:
        """Uploads a checkpoint when exiting the logger."""

    def __enter__(self) -> "Logger":
        logging.info("Starting logger.")
        self._variables_enter = self._get_variables()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self.save_checkpoint:
            self._variables_exit = self._get_variables()
            self._save_and_upload_checkpoint()
        logging.info("Closing logger...")
        self.close()

    def _save_and_upload_checkpoint(self) -> None:
        """Grabs the `training_state` variable from within the context manager, pickles it and
        saves it. This will break if the desired variable to checkpoint is not called
        `training_state`.
        """
        logging.info("Saving checkpoint...")
        in_context_variables = dict(set(self._variables_exit).difference(self._variables_enter))
        variable_id = in_context_variables.get("training_state", None)
        if variable_id is not None:
            training_state = self._variables_exit[("training_state", variable_id)]
        else:
            training_state = None
            logging.debug("Logger did not find variable 'training_state' at the context manager level.")
        with open(self.checkpoint_file_name, "wb") as file_:
            pickle.dump(training_state, file_)
        self.upload_checkpoint()
        logging.info(f"Final Checkpoint saved at '{self.checkpoint_file_name}'.")

    # new
    def _save_and_upload_checkpoint_test(self, training_state, filename) -> None:
        """Grabs the `training_state` variable from within the context manager, pickles it and
        saves it. This will break if the desired variable to checkpoint is not called
        `training_state`.
        """
        logging.info("Saving checkpoint...")
        with open(filename, "wb") as file_:
            pickle.dump(training_state, file_)
        self.upload_checkpoint()
        logging.info(f"Checkpoint saved at '{filename}'.")

    def _get_variables(self) -> Dict:
        """Returns the local variables that are accessible in the context of the context manager.
        This function gets the locals 2 stacks above. Index 0 is this very function, 1 is the
        __init__/__exit__ level, 2 is the context manager level.
        """
        return {(k, id(v)): v for k, v in inspect.stack()[2].frame.f_locals.items()}


class TerminalLogger(Logger):
    """Logs to terminal."""

    def __init__(self, name: Optional[str] = None, save_checkpoint: bool = False):
        super().__init__(save_checkpoint=save_checkpoint)
        if name:
            logging.info(f"Experiment: {name}")

    def _format_values(self, data: Dict[str, Any]) -> str:
        return " | ".join(
            f"{key.replace('_', ' ').title()}: {(f'{value:.3f}' if isinstance(value, float) else f'{value:,}')}"
            for key, value in sorted(data.items())
        )

    def write(self, data: Dict[str, Any], label: Optional[str] = None, env_steps: Optional[int] = None) -> None:
        env_steps_str = f"Env Steps: {env_steps:.2e} | " if env_steps is not None else ""
        label_str = f"{label.replace('_', ' ').title()} >> " if label else ""
        logging.info(label_str + env_steps_str + self._format_values(data))


class TensorboardLogger(Logger):
    """Logs to tensorboard. To view logs, run a command like:
    tensorboard --logdir jumanji/training/outputs/{date}/{time}/{name}/
    """

    def __init__(self, name: str, save_checkpoint: bool = False) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
        if name:
            logging.info(name)
        # log_dir = Path(hydra.core.hydra_config.HydraConfig.get().run.dir + "/" + name)
        self.writer = SummaryWriter(log_dir=name)
        self._env_steps = 0.0

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:
        if env_steps:
            self._env_steps = env_steps
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if np.ndim(metric) == 0:
                if not np.isnan(metric):
                    self.writer.add_scalar(
                        tag=f"{prefix}/{key}",
                        scalar_value=metric,
                        global_step=int(self._env_steps),
                    )
            else:
                raise ValueError(f"Expected metric {key} to be a scalar, got {metric}.")

    def write_hyperparams(self, params: Dict[str, Any]) -> None:

        def flatten_dict(dictionary, parent_key="", sep="_"):
            items = {}
            for key, value in dictionary.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                # if type(value) == dict or type(value) == omegaconf.dictconfig.DictConfig:
                if type(value) == dict:
                    items.update(flatten_dict(value, new_key, sep))
                else:
                    items[new_key] = value
            return items

        params = flatten_dict(params["_content"])

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
        )

    def close(self) -> None:
        self.writer.close()
