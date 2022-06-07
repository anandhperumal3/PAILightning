import warnings
warnings.simplefilter("ignore")
import logging
from functools import partial
import torch
from subprocess import Popen
from lightning.storage import Path
from lightning.components.python import TracerPythonScript
from lightning.components.serve import ServeGradio
from transformers import AutoTokenizer
from pai_datamodule import PAIDataModule
from torch.nn import Softmax

import gradio as gr

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

class TextServeGradio(ServeGradio):

    inputs = gr.inputs.Textbox(lines=2, placeholder="Enter Text Here… ")
    outputs = gr.outputs.Label(num_top_classes=4)

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._labels = {0: 'negative', 1: 'positive', 2: 'no_impact', 3: 'neutral'}

    def run(self, best_model_path):
        self.examples = [ 'with pale blue berries. in these peaceful shades--', 'it flows so long as falls the rain', 'who the man, who, called a brother.']
        self.best_model_path = best_model_path
        super().run()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        print(text)
        pseudonymized = PAIDataModule.synthetic_text({'text': text}, 'text')
        tokenized_txt = tokenizer.encode_plus(pseudonymized['text'], None, add_special_tokens=True, return_token_type_ids=True)
        tokenized_txt = torch.tensor(tokenized_txt['input_ids']).unsqueeze(0)
        prediction = self.model(tokenized_txt)
        softmax = Softmax(dim=1)
        predicted_score = softmax(prediction['logits']).tolist()[0]
        print(predicted_score)
        return {self._labels[i]: predicted_score[i] for i in range(len(predicted_score))}

    def build_model(self):
        model = torch.load(self.best_model_path)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model