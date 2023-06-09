import os
import json

# import numpy as np
import pandas as pd
import cv2 as cv
from skimage import io
# from PIL import Image

from freezegraph import CATEGORIES, DATA_DIR, FROZEN_DIR


def main() -> None:
    # Load image of Lenna:
    # Costs: 3; Element: Water, Type_EN: Forward, Power: 7000
    img = io.imread('https://sakura-pink.jp/img-items/1-ff-2022-12-10-1-7.jpg')
    # img = Image.fromarray(img)
    # img = np.array(img.resize((250, 179), Image.LANCZOS))

    for category in CATEGORIES:
        with open(os.path.join(DATA_DIR, "model", f"{category}_model", "category.json")) as fp:
          classes = json.load(fp)
        
        pb = os.path.join(FROZEN_DIR, f"{category}_model.pb")
        pbt = os.path.join(FROZEN_DIR, f"{category}_model.pbtxt")
        
        # Read the neural network
        cvNet = cv.dnn.readNetFromTensorflow(pb, pbt)

        blob = cv.dnn.blobFromImage(img, size=(250, 179), swapRB=False, crop=False)

        cvNet.setInput(blob)
        
        # Run object detection
        cvOut = cvNet.forward()

        if len(classes) > 2:
          df = pd.Series(cvOut[0], index=classes)
          print(f"{category}: {df.idxmax()}")
        else:
          print(f"{category}: {cvOut[0][0]}")


if __name__ == '__main__':
	main()
