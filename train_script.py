from transformers import AutoModelForSequenceClassification
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics import Accuracy
import torch

from pai_datamodule import PrivateAISynthetic


class SequenceClassification(LightningModule):
    def __init__(self, model=None, data_file=None):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self.model = model or AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=5)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.val_acc = Accuracy()
        self.lr = 0.001
        self.data_file = data_file

    def save_pretrained(self, model_output_dir):
        torch.save(self.model, model_output_dir)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, _, labels = batch
        output = self.forward(input_ids)
        loss = self.loss_function(output['logits'], labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, _, labels = batch
        output = self.forward(input_ids)
        loss = self.loss_function(output['logits'], labels)
        self.log("val_acc", self.val_acc(output['logits'], labels), on_step=True, on_epoch=True)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, )


if __name__ == "__main__":
    cli = LightningCLI(
        SequenceClassification, PrivateAISynthetic, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
