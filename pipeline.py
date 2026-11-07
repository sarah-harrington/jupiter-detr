
import torch
import argparse
import os
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class DETR_Model:

    def __init__(self):
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd")
        self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")

        self.data_path = ""

    def get_dataset(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
        args = parser.parse_args()
        return args.data_path

    def process_dataset(self):
        # check path is valid and is a directory
        if os.path.exists(self.data_path) and os.path.isdir(self.data_path):
            print(f"Dataset path {self.data_path} is valid.")

            # confirm dataset isn't empty, then check for valid image files
            with os.scandir(self.data_path) as entries:
                if not any(entries):
                    raise ValueError(f"Dataset path {self.data_path} is empty.")
                else: 
                    print(f"Dataset path {self.data_path} is not empty. Checking for valid image files...")
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                            print(f"Found valid image file: {entry.name}. Checking if it is corrupted...")
                            # check for corrupted images
                            try:
                                with Image.open(entry.path) as img:
                                    # image verify returns None if the image is valid, otherwise it raises an exception
                                    if img.verify() == None:
                                        print(f"Image file {entry.name} is valid.")
                                        return True 
                                    else:
                                        print(f"Image file {entry.name} could not be verified.")
                            except (IOError, SyntaxError) as e:
                                raise ValueError(f"Image file {entry.name} is corrupted or unreadable: {e}")

                        else:
                            raise ValueError(f"Dataset path {self.data_path} does not contain any valid image files.")
        else:
            raise ValueError(f"Dataset path {self.data_path} is not valid.")



    def predict(self, image_path):

        image = Image.open(image_path)
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Get the predicted bounding boxes and labels
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        return results


if __name__ == "__main__":
    model_object = DETR_Model()

    # get dataset path from command line arguments  
    model_object.data_path = model_object.get_dataset()

    # check dataset
    model_object.process_dataset()