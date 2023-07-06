import os
import cv2
import numpy as np
from natsort import natsorted

from mltu.inferenceModel import OnnxInferenceModel
from mltu.text_utils import ctc_decoder

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    # Load the model configurations
    configs = BaseModelConfigs.load("Models/202307061625/configs.yaml")

    # Create the ImageToWordModel instance
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    # Specify the folder containing preprocessed images
    folder_path = "DataTest/preprocessed"

    # Get a sorted list of image files in the folder using natural sorting
    image_files = natsorted(os.listdir(folder_path))

    # Specify the output file to store the predictions
    output_file = "predictions.txt"

    with open(output_file, "w") as f:
        # Write the header row
        f.write("imagefile,prediction_text\n")

        # Iterate through each image file in the sorted order
        for image_file in tqdm(image_files):
            # Read the image
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)

            try:
                prediction_text = model.predict(image)

                # Write the prediction to the output file
                f.write(f"{image_file},{prediction_text}\n")

            except:
                continue
