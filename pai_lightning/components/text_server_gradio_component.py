import warnings
warnings.simplefilter("ignore")

import json
import requests

import torch
from transformers import AutoTokenizer
from torch.nn import Softmax
import gradio as gr

from lightning.app.components.serve import ServeGradio


class TextServeGradio(ServeGradio):

    inputs = gr.inputs.Textbox(lines=2, placeholder="Enter Text Here… ")
    outputs = gr.outputs.Label(num_top_classes=5)

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None
        self.private_ai_synthetic_data = None
        self.pai_port = None
        self.pai_host = None
        self._labels = {0: '0', 1: '1', 2: '2', 3: '3', 4:'4'}

    def run(self, best_model_path, pai_host, pai_port):
        self.examples = [ 'with pale blue berries. in these peaceful shades--', 'it flows so long as falls the rain', 'who the man, who, called a brother.']
        self.best_model_path = best_model_path
        self.pai_host = pai_host
        self.pai_port = pai_port
        super().run()

    def request_call(self, text):
        payload = json.dumps({
            "text": text,
            "key": "INTERNAL_TESTING_UNLIMITED_REALLY",
            "fake_entity_accuracy_mode": "standard"
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", f"http://{self.pai_host}:{self.pai_port}/deidentify_text", headers=headers, data=payload)
        pseudonymized_text = response.json()["result_fake"]
        return pseudonymized_text

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        pseudonymized_text = self.request_call(text)
        tokenized_txt = tokenizer.encode_plus(pseudonymized_text, None, add_special_tokens=True,
                                              return_token_type_ids=True)
        tokenized_txt = torch.tensor(tokenized_txt['input_ids']).unsqueeze(0)
        prediction = self.model(tokenized_txt)
        softmax = Softmax(dim=1)
        predicted_score = softmax(prediction['logits']).tolist()[0]
        return {self._labels[i]: predicted_score[i] for i in range(len(predicted_score))}

    def build_model(self):
        model = torch.load(self.best_model_path)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model
