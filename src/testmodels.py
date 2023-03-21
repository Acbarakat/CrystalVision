# -*- coding: utf-8 -*-
"""
Test models against real-world (internet data)

Attributes:
    CATEGORIES (tuple): A tuple of the types of ``models`` trained
	IMAGES (np.ndarray): A list or URIs converted to loaded and resized
		image data
	DF (pd.DataFrame): A dataframe full of accurate card data which
		we will compare model predictions against

Todo:
    * Find more varied data, such as off-center or card with background

"""
import os
import json

import pandas as pd
import tensorflow as tf
import numpy as np
from skimage.io import imread
import keras

from gatherdata import DATA_DIR


CATEGORIES = ("Element", "Type_EN", "Cost", "Power")

IMAGES = [
	'https://sakura-pink.jp/img-items/1-ff-2022-12-10-1-7.jpg',  # JP - Lenna
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1107073.jpg',  # Foil - Seph
	'https://assets-prd.ignimgs.com/2022/10/26/18-050l-2-eg-1666822378783.jpg',  # Premium - Yuffie FA
	'https://www.legendarywolfgames.com/wp-content/uploads/2020/03/FFTCG-PR072.jpg',  # Premium - Chelinka
	'http://img.over-blog-kiwi.com/2/21/23/79/20181209/ob_78f42c_y-shtola.jpg',  # FA F Chinese - Yshtola
	'https://i.ebayimg.com/images/g/YuAAAOSwZxNjfXJn/s-l1600.jpg', # F Ifrit
	'https://d1rw89lz12ur5s.cloudfront.net/photo/bigncollectibles/file/1407913/5.jpg?1518673189', # F Nono
	'https://crystal-cdn4.crystalcommerce.com/photos/6476018/7-089foil.jpg', # F Coeurl
	'https://crystal-cdn2.crystalcommerce.com/photos/6479730/7-098foil.jpg', # F Flanbord
	'https://i.ebayimg.com/images/g/A7EAAOSwQ0JhgY90/s-l500.jpg', # F Shantotto Backup
	# 'https://pbs.twimg.com/media/FhOOgabagAEbuya.jpg'
	# 'http://fftcg.cdn.sewest.net/images/cards/full/B-015_eg.jpg'
	# 'https://pbs.twimg.com/media/FgrQ-fTaYAIkguF.jpg'
	# 'https://crystalcommerce-assets.s3.amazonaws.com/photos/6548938/large/11-136s-fl-eg.jpg?1581219705'
	# 'https://product-images.tcgplayer.com/211239.jpg'
]
IMAGES = [imread(image) for image in IMAGES]
IMAGES = np.array([tf.image.resize(image, (250, 179)) for image in IMAGES])

DF = pd.DataFrame(
	columns=["Code", "Name_EN", "Element", "Type_EN", "Cost", "Power"],
	data=[
		("18-100L", "Lenna", "\u6c34", "Forward", "3", "7000"),
		("1-044R", "Sephiroth", "\u6c37", "Forward", "5", "7000"),
		("18-050L", "Yuffie", "\u98a8", "Forward", "5", "9000"),
		("7-054L", "Chelinka", "\u98a8", "Forward", "3", "7000"),
		("5-068L", "Yshtola", "\u98a8", "Forward", "3", "7000"),
		("1-004C", "Ifrit", "\u706b", "Summon", "1", ""),
		("4-066R", "Nono", "\u98a8", "Backup", "3", ""),
		("7-089C", "Coeurl", "\u96f7", "Monster", "2", ""),
		("7-098R", "Flanborg", "\u96f7", "Monster", "2", "7000"),
		("1-107L", "Shantotto", "\u571f", "Backup", "7", "")
	]
)


def main() -> None:
	for category in CATEGORIES:
		model_path = os.path.join(DATA_DIR, "model", f"{category}_model")

		with open(os.path.join(model_path, "category.json")) as fp:
			labels = json.load(fp)

		model = keras.models.load_model(model_path)

		x = model.predict(IMAGES)
		DF[f"predict_{category}"] = [labels[np.argmax(y)] for y in x]

		comp = DF[category] == DF[f"predict_{category}"]
		comp = comp.value_counts(normalize=True)

		print(f"{category} accuracy: {comp[True] * 100}%%")
		# print(x)

	print(DF)


if __name__ == '__main__':
	main()
