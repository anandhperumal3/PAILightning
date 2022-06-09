import time
from subprocess import Popen
import json, os
from pathlib import Path
import lightning as L
import requests
from datasets import load_dataset


class PrivateAISyntheticData(L.LightningWork):

    LOCAL_STORE_DIR = Path(os.path.join(Path.cwd(), ".lightning-store"))
    if not Path.exists(LOCAL_STORE_DIR):
        Path.mkdir(LOCAL_STORE_DIR)
    def __init__(self, key, mode, text_feature, output_path, host, port):
        """
        Private-AI Data Module, for synthetic data generation
        :param key: PAI customer Key
        :param text_features: list of text feature names in the dataset that needs a synthetic data generation
        :param mode: synthetic data generation model type
        :param url: docker url

        """
        super().__init__(host=host, port=port)
        self.key = key
        self.mode = mode
        self.text_feature = text_feature
        self.url = None
        self.server_started = False
        self.output_path = str(Path(
            os.path.join(
                self.LOCAL_STORE_DIR,
                output_path,
            )
        ))

    def start_server(self, host, port):
        # start docker
        cmd = f"docker run --rm -p {port}:{port} deid:2.11full"
        if not self.url:
            self.url = f"http://{host}:{port}/deidentify_text"

        # Popen(cmd.split(" "))
        # time.sleep(600)

        return

    def pai_docker_call(self, text) -> str:
        """
        This function makes a call to the PAI docker for the synthetic text generation.
        :param text: input text
        :return: synthetic text
        """
        payload = json.dumps({
            "text": text,
            "key": self.key,
            "fake_entity_accuracy_mode": self.mode
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        fake_text = response.json()["result_fake"]
        return fake_text

    def synthetic_text(self, example, text_feature_name) -> object:
        """
        This function modifies the text in the example with synthetic text.
        :param example: example containing the text column as a field
        :param text_feature_name: text feature name
        :return: example containing the synthetic text in the text feature field
        """
        example[text_feature_name] = self.pai_docker_call(example[text_feature_name])
        return example

    def run(self, input_text_or_path: str, action: str = 'individual'):
        if not self.server_started:
            self.start_server(self.host, self.port)
            self.server_started = True
        if action == 'batch':
            synthetic_data = load_dataset('csv', data_files=input_text_or_path)
            synthetic_data = synthetic_data.map(lambda e: self.synthetic_text(e, self.text_feature))
            if not os.path.exists(os.path.dirname(self.output_path)):
                os.mkdir(os.path.dirname(self.output_path))
            synthetic_data['train'].to_csv(self.output_path)
            self.output_path =  L.storage.Payload(self.output_path)
        else:
            synthetic_data = self.synthetic_text({'text': input_text_or_path}, self.text_feature)
            return synthetic_data['text']
        return None