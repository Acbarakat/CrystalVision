import os
import json

import pandas as pd
import cv2 as cv
from skimage import io

from freezegraph import CATEGORIES, DATA_DIR


FROZEN_DIR = os.path.join(DATA_DIR, "frozen")


def main() -> None:
    # Load image of Lenna:
    # Costs: 3; Element: Water, Type_EN: Forward, Power: 7000
    img = io.imread('https://sakura-pink.jp/img-items/1-ff-2022-12-10-1-7.jpg')

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

        df = pd.Series(cvOut[0], index=classes)
        print(f"{category}: {df.idxmax()}")


if __name__ == '__main__':
	main()
