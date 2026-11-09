
import torch
import argparse
import os
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, Trainer, TrainingArguments

class DETR_Model:

    def __init__(self):
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd")
        self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd")

        self.data_path = ""
        self.image_files_list = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

    def get_dataset(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
        args = parser.parse_args()
        return args.data_path

    def process_dataset(self):
        # first confirm directory and image files are valid
        if self.check_dataset_directory():
            print("Dataset directory check completed successfully.")

            # if dataset and image files are valid, proceed to image preprocessing
            self.preprocess_images()
        else:
            print("ERROR:Dataset directory check failed.")
            exit()


    def check_dataset_directory(self):
        # check path is valid and is a directory
        if os.path.exists(self.data_path) and os.path.isdir(self.data_path):
            print(f"Dataset path {self.data_path} is valid and is a directory.")

            entries = list(os.scandir(self.data_path))  # Convert to list to allow multiple iterations
            
            # confirm dataset isn't empty, then check for valid image files:
            if not entries:
                raise ValueError(f"Dataset directory {self.data_path} is empty.")
            else: 
                print(f"Dataset directory {self.data_path} is not empty. Checking for valid image files...")
          
            # have to keep iterating through entries separate from scandir
            print(f"Number of entries in dataset: {len(list(entries))}")

            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"Found valid image file: {entry.name}. Checking if it is corrupted...")
                    # check for corrupted images
                    try:
                        with Image.open(entry.path) as img:
                            # image verify returns None if the image is valid, otherwise it raises an exception
                            if img.verify() == None:
                                print(f"Image file {entry.name} is valid.")

                                # add opened image to list for preprocessing
                                self.image_files_list.append(entry)
                            else:
                                print(f"Image file {entry.name} could not be verified.")
                                return false

                        img.close()  # Close the image file after verification
                    except (IOError, SyntaxError) as e:
                        raise ValueError(f"Image file {entry.name} is corrupted or unreadable: {e}")

                else:
                    print(f"Found non-image file: {entry.name}. Skipping...")
        else:
            raise ValueError(f"Dataset path {self.data_path} is not valid.")

        return True


    def preprocess_images(self):
        print("Starting image preprocessing...")

        # Iterate through the list of image files and preprocess each one
        for image in self.image_files_list:
            print(f"Preprocessing image: {image.name}")

            # placeholder for image to save after preprocessing
            processed_image = image

            #reopen image
            opened_image = Image.open(image.path)

            # resize as needed
            print(f"Image size before preprocessing: {opened_image.size}")

            if opened_image.size[0] != 640 or opened_image.size[1] != 640:
                opened_image = opened_image.resize((640, 360))
                print(f"Image size after resizing: {opened_image.size}")

            # normalize pixels for RT DETR


    def predict(self, image_path):
        
        
        inputs = self.image_processor(images=, return_tensors="pt")
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