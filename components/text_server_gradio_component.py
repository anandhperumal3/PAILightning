import warnings
warnings.simplefilter("ignore")
import logging
import json, requests
import torch

from lightning.components.serve import ServeGradio
from transformers import AutoTokenizer
from torch.nn import Softmax

import gradio as gr

logger = logging.getLogger(__name__)


class TextServeGradio(ServeGradio):

    inputs = gr.inputs.Textbox(lines=2, placeholder="Enter Text Here… ")
    outputs = gr.outputs.Label(num_top_classes=5)

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None
        self._labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'}

    def run(self, best_model_path):
        self.examples = [ 'with pale blue berries. in these peaceful shades--', 'it flows so long as falls the rain', 'who the man, who, called a brother.']
        self.best_model_path = best_model_path
        super().run()

    def predict(self, text, url=None,mode=None, key=None):
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

        payload = json.dumps({
            "text": text,
            "key": "XXX",
            "fake_entity_accuracy_mode": "standard"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", "http://localhost:8080/deidentify_text", headers=headers, data=payload)
        pseudonymized_text = response.json()["result_fake"]

        tokenized_txt = tokenizer.encode_plus(pseudonymized_text, None, add_special_tokens=True, return_token_type_ids=True)
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
