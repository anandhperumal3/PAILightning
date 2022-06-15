import lightning as L
from lightning.storage import Drive

from pai_lightning.components import PrivateAISyntheticData, PyTorchLightningScript, TextServeGradio

import os


class TrainDeploy(L.LightningFlow):
    def __init__(self, host, port):
        super().__init__()
        self.input_path = "./datasets/data.csv"
        self.output_path = "./datasets/data_output.csv"
        self.key = "INTERNAL_TESTING_UNLIMITED_REALLY"
        self.mode = "standard"
        self.drive = Drive("lit://private_ai_app")
        self.private_ai_synthetic_data = PrivateAISyntheticData(key=self.key, mode=self.mode, 
                                                                text_features="text",
                                                                host=host, port=port, drive=self.drive,
                                                                output_path=self.output_path)

        self.train_work = PyTorchLightningScript(
            script_path=os.path.join(os.path.dirname(__file__), "./pai_lightning/scripts/train_script.py"),
            script_args=["--trainer.max_epochs=1", f"--data.data_file={self.private_ai_synthetic_data.output_path}",
                         "--data.text_feature=text"],
        )

        self.serve_work = TextServeGradio(cloud_compute=L.CloudCompute("cpu", 1))

    def run(self):
        self.private_ai_synthetic_data.run(input_path=self.input_path)

        # 1. Run the python script that trains the model
        self.train_work.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path, pai_host="127.0.0.1", pai_port=8080)

    def configure_layout(self):
        tab_1 = {"name": "Model training", "content": self.train_work}
        tab_2 = {"name": "Interactive demo", "content": self.serve_work}
        return [tab_1, tab_2]

app = L.LightningApp(TrainDeploy(host="localhost", port="8080"))
