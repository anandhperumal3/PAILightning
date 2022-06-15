import warnings
warnings.simplefilter("ignore")

import logging
from functools import partial
import torch

from lightning.storage import Path
from lightning.components.python import TracerPythonScript

from subprocess import Popen
logger = logging.getLogger(__name__)


class PyTorchLightningScript(TracerPythonScript):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, raise_exception=True, **kwargs)
        self.best_model_path = None
        self._process = None

    def configure_tracer(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import Callback

        tracer = super().configure_tracer()

        class CollectURL(Callback):
            def __init__(self, work):
                self._work = work

            def on_train_start(self, trainer, *_):
                cmd = f"tensorboard --logdir={trainer.logger.log_dir} --host {self._work.host} --port {self._work.port}"
                self._work._process = Popen(cmd.split(" "))

        def trainer_pre_fn(self, *args, work=None, **kwargs):
            kwargs['callbacks'].append(CollectURL(work))
            return {}, args, kwargs

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=partial(trainer_pre_fn, work=self))
        return tracer

    def run(self, *args, **kwargs):
        self.script_args += [
            "--trainer.limit_train_batches=4",
            "--trainer.limit_val_batches=4",
            "--trainer.callbacks=ModelCheckpoint",
            "--trainer.callbacks.monitor=val_acc",
        ]
        warnings.simplefilter("ignore")
        logger.info(f"Running train_script: {self.script_path}")
        super().run(*args, **kwargs)

    def on_after_run(self, res):
        lightning_module = res["cli"].trainer.lightning_module
        checkpoint = torch.load(res["cli"].trainer.checkpoint_callback.best_model_path)
        lightning_module.load_state_dict(checkpoint["state_dict"])
        lightning_module.save_pretrained("pytorch_model.bin")
        self.best_model_path = Path("pytorch_model.bin")
