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
import glob

import pandas as pd
import tensorflow as tf
import numpy as np
from skimage.io import imread
from PIL import Image
from mlxtend.classifier import EnsembleVoteClassifier

from gatherdata import DATA_DIR


CATEGORIES = ("Element", "Type_EN", "Cost", "Power", "Ex_Burst")

IMAGES = [
	'https://sakura-pink.jp/img-items/1-ff-2022-12-10-1-7.jpg',	 # JP - Lenna
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1107073.jpg',	# Foil - Seph
	'https://assets-prd.ignimgs.com/2022/10/26/18-050l-2-eg-1666822378783.jpg',	 # Premium - Yuffie FA
	'https://www.legendarywolfgames.com/wp-content/uploads/2020/03/FFTCG-PR072.jpg',  # Premium - Chelinka
	'http://img.over-blog-kiwi.com/2/21/23/79/20181209/ob_78f42c_y-shtola.jpg',	 # FA F Chinese - Yshtola
	'https://i.ebayimg.com/images/g/YuAAAOSwZxNjfXJn/s-l1600.jpg', # F Ifrit
	'https://d1rw89lz12ur5s.cloudfront.net/photo/bigncollectibles/file/1407913/5.jpg', # F Nono
	'https://crystal-cdn4.crystalcommerce.com/photos/6476018/7-089foil.jpg', # F Coeurl
	'https://crystal-cdn2.crystalcommerce.com/photos/6479730/7-098foil.jpg', # F Flanbord
	'https://i.ebayimg.com/images/g/A7EAAOSwQ0JhgY90/s-l500.jpg', # F Shantotto Backup
	'https://cdn.shopify.com/s/files/1/1715/6019/products/e1d86d4e-7a92-48e7-ab55-73f16c2787e1_600x.png', # F Ark
	'https://i.ebayimg.com/images/g/WWAAAOSweGNhHgtE/s-l500.jpg', # F Chaos
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Materia-13-103L-OpusXIII-Foil_500x.png', # F Materia
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1167486.jpg', # F Aleoidai
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1661117.jpg', # F Man in Black
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1146230.jpg', # F Cactuar
	'https://superelf-cards.de/storage/images/image?remote=https%3A%2F%2Fsuperelf-cards.de%2FWebRoot%2FStore6%2FShops%2FShop052035%2F5F3A%2F7ABF%2FF0F3%2F877E%2F41BF%2FAC14%2F500D%2F36D8%2F6-074F.jpg', # F German Cactuar
	'https://i.ebayimg.com/images/g/e60AAOSwerdand5v/s-l500.jpg', # Foil Kain on a big black background
	'http://cdn.shopify.com/s/files/1/1715/6019/products/AsheFoil_1024x.jpg', # Foil Ashe
	'https://cdn.shopify.com/s/files/1/0988/3294/products/fftcg_fftcgpr0264037hfoil_1_1024x1024.jpg', # Foil JP Serah Promo
	'https://i.ebayimg.com/images/g/rhQAAOSwEFxh5UsR/s-l500.jpg', # Foil light Cloud EX
	'https://crystal-cdn2.crystalcommerce.com/photos/6479590/7-045foil.jpg', # Foil Alexander EX
	'https://crystal-cdn2.crystalcommerce.com/photos/6479652/7-031foil.jpg', # Foil Shiva EX
	'https://cdn.shopify.com/s/files/1/1715/6019/products/CidofClanGullyEX-17-094CFoil_500x.png', # Foil Cid of Clan Gully EX
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1114146.jpg', # F Merlwyb EX
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1167552.jpg', # F Star Subyl Ex
	'https://cdn.shopify.com/s/files/1/1715/6019/products/AsheEXFullArt_Foil_500x.png', # F Full Art Ashe EX
	'https://i.ebayimg.com/images/g/fnMAAOSwaHdhcz6-/s-l500.jpg', # F JP Laguna
	'https://cdn.shopify.com/s/files/1/1715/6019/products/EwenEX-17-080RFoil_600x.png', # F Ewen
	'https://52f4e29a8321344e30ae-0f55c9129972ac85d6b1f4e703468e6b.ssl.cf2.rackcdn.com/products/pictures/1612601.jpg', # F Terra EX
	# 'https://pbs.twimg.com/media/FhOOgabagAEbuya.jpg'
	# 'http://fftcg.cdn.sewest.net/images/cards/full/B-015_eg.jpg'
	# 'https://pbs.twimg.com/media/FgrQ-fTaYAIkguF.jpg'
	# 'https://crystalcommerce-assets.s3.amazonaws.com/photos/6548938/large/11-136s-fl-eg.jpg?1581219705'
	# 'https://product-images.tcgplayer.com/211239.jpg'
	# 'https://cdn.shopify.com/s/files/1/1715/6019/products/Akstar_Foil_600x.png',
]

IMAGE_DF = pd.DataFrame(
	columns=["Code", "Name_EN", "Element", "Type_EN", "Cost", "Power", "Ex_Burst"],
	data=[
		("18-100L", "Lenna", "\u6c34", "Forward", "3", "7000", 0),
		("1-044R", "Sephiroth", "\u6c37", "Forward", "5", "7000", 0),
		("18-050L", "Yuffie", "\u98a8", "Forward", "5", "9000", 0),
		("7-054L", "Chelinka", "\u98a8", "Forward", "3", "7000", 0),
		("5-068L", "Yshtola", "\u98a8", "Forward", "3", "7000", 0),
		("1-004C", "Ifrit", "\u706b", "Summon", "1", "", 1),
		("4-066R", "Nono", "\u98a8", "Backup", "3", "", 0),
		("7-089C", "Coeurl", "\u96f7", "Monster", "2", "", 0),
		("7-098R", "Flanborg", "\u96f7", "Monster", "2", "7000", 0),
		("1-107L", "Shantotto", "\u571f", "Backup", "7", "", 0),
		("8-135H", "Ark", "\u95c7", "Summon", "10", "", 0),
		("1-184H", "Chaos", "\u95c7", "Backup", "2", "", 0),
		("13-103L", "Materia", "\u5149", "Forward", "1", "2000", 0),
		("5-025H", "Aleoidai", "\u6c37", "Monster", "4", "", 0),
		("11-093H", "Man in Black", "\u96f7", "Forward", "5", "9000", 0),
		("4-058C", "Cactuar", "\u98a8", "Monster", "1", "", 0),
		("6-074C", "Cactuar", "\u571f", "Summon", "4", "", 0),
		("2-103H", "Kain", "\u96f7", "Forward", "3", "5000", 0),
		("12-037L", "Ashe", "\u98a8", "Forward", "2", "5000", 0),
		("PR-026", "Serah", "\u6c37", "Forward", "2", "5000", 0),
		("4-145H", "Cloud", "\u5149", "Forward", "3", "7000", 1),
		("7-045C", "Alexander", "\u98a8", "Summon", "2", "", 1),
		("7-031C", "Shiva", "\u6c37", "Summon", "3", "", 1),
		("17-094C", "Cid of Clan Gully", "\u96f7", "Backup", "4", "", 1),
		("2-137H", "Merlwyb", "\u6c34", "Backup", "4", "", 1),
		("5-091H", "Star Sibyl", "\u571f", "Backup", "5", "", 1),
		("18-086H", "Ashe", "\u6c34", "Forward", "6", "9000", 1),
		("1-059R", "Laguna", "\u6c37", "Forward", "4", "7000", 1),
		("17-080R", "Ewen", "\u571f", "Forward", "1", "3000", 1),
		("10-132S", "Terra", "\u706b", "Forward", "4", "5000", 1),
	]
)


class MyEnsembleVoteClassifier(EnsembleVoteClassifier):
	def __init__(self, clfs, voting="hard", weights=None, verbose=0, use_clones=True, fit_base_estimators=True):
		super().__init__(clfs, voting, weights, verbose, use_clones, fit_base_estimators)
		self.clfs_ = clfs

	def _predict(self, X: np.ndarray) -> None:
		"""Collect results from clf.predict calls."""

		if not self.fit_base_estimators:
			return np.asarray([clf(X) for clf in self.clfs_]).T
		else:
			return np.asarray(
				[self.le_.transform(clf(X)) for clf in self.clfs_]
			).T

	def predict(self, X: np.ndarray, dtype: str='int64') -> np.ndarray:
		"""Predict class labels for X.

		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.

		Returns
		----------
		maj : array-like, shape = [n_samples]
			Predicted class labels.

		"""
		if self.voting == "soft":
			maj = np.argmax(self.predict_proba(X), axis=1)

		else:  # 'hard' voting
			predictions = self._predict(X).astype(dtype)[0]

			maj = np.apply_along_axis(
				lambda x: np.argmax(np.bincount(x, weights=self.weights)),
				axis=1,
				arr=predictions,
			)

		if self.fit_base_estimators:
			maj = self.le_.inverse_transform(maj)

		return maj

	# TODO: Save ensemble in keras format
	def save_model():
		raise NotImplementedError()


def load_image(url: str,
			   img_fname: str='') -> np.ndarray:
	'''
	Load image (and cache it)

	Args:
		url (str): The image URL
		img_fname (str): The file name we will save to
			(default will be derived from url)
	
	Returns:
		Image as ndarray
	'''
	if not img_fname:
		img_fname = url.split("/")[-1]
		img_fname = img_fname.split("%2F")[-1]

	dst = os.path.join(DATA_DIR, "test")
	if not os.path.exists(dst):
		os.makedirs(dst)

	dst = os.path.join(dst, img_fname)
	if os.path.exists(dst):
		return imread(dst)[:,:,:3]

	data = imread(url)
	if img_fname.endswith(".jpg"):
		Image.fromarray(data).convert("RGB").save(dst)
	else:
		Image.fromarray(data).save(dst)

	return data[:,:,:3]


IMAGES = [load_image(image, f"{idx}.jpg") for idx, image in enumerate(IMAGES)]	# Drop Alpha channels
IMAGES = np.array([tf.image.resize(image, (250, 179)) for image in IMAGES])


def test_models() -> pd.DataFrame:
	'''
	Run all models and apply values in the datagrame
	
	Returns:
		ImageData dataframe with yhat(s)
	'''
	df = IMAGE_DF.copy()

	for category in CATEGORIES:
		model_path = os.path.join(DATA_DIR, "model", f"{category}_model")

		with open(os.path.join(model_path, "category.json")) as fp:
			labels = json.load(fp)

		if category == "Ex_Burst":
			models = [tf.keras.models.load_model(model_path) for model_path in glob.iglob(model_path + "*")]
			voting = MyEnsembleVoteClassifier(models, fit_base_estimators=False) #, weights=[0, 1, 0, 0, 0])
			x = voting.predict(IMAGES)
		else:
			model = tf.keras.models.load_model(model_path)
			x = model(IMAGES, training=False)

		if len(labels) > 2:
			# xf = pd.DataFrame(x, columns=labels)
			df[f"{category}_yhat"] = [labels[np.argmax(y)] for y in x]
		else:
			df[f"{category}_yhat"] = np.round(x)
			df[f"{category}_yhat"] = df[f"{category}_yhat"].astype('UInt8')

	return df


def main() -> None:
	df = test_models()

	for category in CATEGORIES:
		comp = df[category] == df[f"{category}_yhat"]
		comp = comp.value_counts(normalize=True)

		print(f"{category} accuracy: {comp[True] * 100}%%")
		# print(xf)

	df.sort_index(axis=1, inplace=True)
	print(df)


if __name__ == '__main__':
	main()
