# TODO: imports for docker
import time
from subprocess import Popen
import json
import requests

import lightning as L
from lightning.app.storage import Drive

from datasets import load_dataset

import os
from typing import List, Union


# class DockerBuildConfig(L.BuildConfig):
#     def build_commands(self):
#         return [
#         ]


class PrivateAISyntheticData(L.LightningWork):
    def __init__(
        self,
        key: str,
        mode: str,
        text_features: Union[List[str], str],
        drive: Drive,
        output_path: str
    ):
        """
        Private-AI Data Module, for synthetic data generation

        :param key: PAI customer Key
        :param mode: synthetic data generation model type
        :param text_features: list/str of text feature names in the dataset that needs a synthetic data generation
        # :param host: host address (str) for the Work to be started at
        # :param port: port address (int) for the Work to be started at
        :param drive: a lightning.storage.Drive object, where the data exchange will take place
        :param output_path: The relative path (including filename + extension) where you want to store the synthetically generated file
        """
        super().__init__(
            cloud_build_config=L.BuildConfig(
                image="gcr.io/grid-backend-266721/private_ai:latest", 
            )
        )

        self.key = key
        self.mode = mode
        if isinstance(text_features, str):
            text_features = [text_features]
        self.text_features = text_features
        self.url = None

        # TODO: used for docker
        # self.server_started = False

        self.drive = drive
        self.output_path = output_path

    # def start_server(self):
    #     # start docker
    #     # cmd = f"docker run --rm -p {port}:{port} gcr.io/grid-backend-266721/private_ai:latest"
    #     if not self.url:
    #         self.url = f"http://{self.host}:{self.port}/deidentify_text"
    #
    #     Popen(cmd.split(" "))
    #     time.sleep(600)
    #
    #     return

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
        self.url = f"http://{self.host}:{self.port}/deidentify_text"
        response = requests.request("POST", self.url, headers=headers, data=payload)
        fake_text = response.json()["result_fake"]
        return fake_text

    def synthetic_text(self, example, text_feature_names: List[str]) -> object:
        """
        This function modifies the text in the example with synthetic text.

        :param example: example containing the text column as a field
        :param text_feature_names: text feature name(s)
        :return: example containing the synthetic text in the text feature field
        """
        for text_feature_name in text_feature_names:
            example[text_feature_name] = self.pai_docker_call(example[text_feature_name])
        # TODO: This is temporary fix, remove the line below once docker calls are tested and are working
        # for feat in text_feature_names:
        #     if feat in example:
        #         example[feat] = "fake"
        # example["label"] = 1
        return example

    def run(self, input_path: str):
        # if not self.server_started:
        #     # self.start_server()
        #     self.server_started = True

        # Attempt to get the input file path to the local file system (if it doesn't exist already!!)
        if not os.path.exists(input_path):
            self.drive.get(input_path)

        synthetic_data = load_dataset('csv', data_files=input_path)
        synthetic_data = synthetic_data.map(lambda row: self.synthetic_text(row, self.text_features))
        synthetic_data['train'].to_csv(self.output_path)
        self.drive.put(self.output_path)
