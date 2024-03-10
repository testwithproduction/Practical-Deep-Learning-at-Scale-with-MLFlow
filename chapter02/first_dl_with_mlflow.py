import flash
import mlflow
import torch
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

print("### download IMDb data to local folder")
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    batch_size=4,    
)

print("### define a text classifier ")
classifier_model = TextClassifier(
    backbone="prajjwal1/bert-tiny", labels=datamodule.labels, num_classes=datamodule.num_classes
)

print("### define the trainer")
trainer = flash.Trainer(max_epochs=4, accelerator="mps", devices=1)

EXPERIMENT_NAME = "dl_model_chapter02"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

mlflow.pytorch.autolog()

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="chapter02"):
    trainer.finetune(classifier_model, datamodule=datamodule, strategy="freeze")
    # trainer.test()
