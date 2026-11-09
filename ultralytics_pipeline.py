from ultralytics import RTDETR
import argparse

class Ultralyics_DETR_Pipeline:
    def __init__(self):
        self.model = RTDETR("rtdetr-1.pt")

        self.epochs = 1
        self.img_size = 640

        self.dataset_yaml_path = None

    def get_dataset(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
        args = parser.parse_args()
        self.dataset_yaml_path = args.data_path

    def train(self):
        self.model.train(
            data=self.dataset_yaml_path,
            epochs=self.epochs,
            imgsz=self.img_size,
        )

    def validate(self):
        self.model.val(
            data=self.dataset_yaml_path,
            imgsz=self.img_size,
        )

    def predict(self):
        self.model.predict(
            source=self.dataset_yaml_path,
            imgsz=self.img_size,
        )



if __name__ == "__main__":

    pipeline = Ultralyics_DETR_Pipeline()
    pipeline.get_dataset()

    pipeline.train()