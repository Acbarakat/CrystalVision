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
from generatemodels import make_database


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
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Akstar_Foil_600x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/SephirothFoil_1024x.jpg',
	'https://i.ebayimg.com/images/g/rTkAAOSw7wljo1MD/s-l500.jpg',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/ShantottoFoil_1024x.jpg',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Y_shtolaFoil_1024x.jpg',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Firion_Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Arciela_Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Relm-19-127L-From-Nightmares-Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Prishe-19-111L-From-Nightmares-Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/SononEX_Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Golbez-13-115L-OpusXIII-Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/NoctisEX_Foil_500x.png',
	'https://cdn.shopify.com/s/files/1/1715/6019/products/Vaan-EX-19-107C-From-Nightmares-Foil_500x.png',
]

IMAGE_DF = pd.DataFrame(
	columns=["Code", "Full_Art", "Foil"],
	data=[
		("18-100L", 0, 0),
		("1-044R", 0, 1),
		("18-050L", 1, 0),
		("7-054L", 1, 0),
		("5-068L", 1, 1),
		("1-004C", 0, 1),
		("4-066R", 0, 1),
		("7-089C", 0, 1),
		("7-098R", 0, 1),
		("1-107L", 0, 1),
		("8-135H", 0, 1),
		("1-184H", 0, 1),
		("13-103L", 0, 1),
		("5-025H", 0, 1),
		("11-093H", 0, 1),
		("4-058C", 0, 1),
		("6-074C", 0, 1),
		("2-103H", 0, 1),
		("12-037L", 0, 1),
		("4-037H", 0, 1),  # PR-026/4-037H
		("4-145H", 0, 1),
		("7-045C", 0, 1),
		("7-031C", 0, 1),
		("17-094C", 0, 1),
		("2-137H", 0, 1),
		("5-091H", 0, 1),
		("18-086H", 1, 1),
		("1-059R", 0, 1),
		("17-080R", 0, 1),
		("10-132S", 0, 1),
		("18-107L", 0, 1),
		("14-123C", 0, 1),
		("18-116L", 0, 1),
		("12-120C", 0, 1),
		("12-119L", 0, 1),
		("18-130L", 0, 1),
		("18-128H", 0, 1),
		("19-127L", 0, 1),
		("19-111L", 0, 1),
		("18-123L", 0, 1),
		("13-115L", 0, 1),
		("18-139S", 0, 1),
		("19-107C", 0, 1)
	]
)


class MyEnsembleVoteClassifier(EnsembleVoteClassifier):
	def __init__(self, clfs, voting="hard", weights=None, verbose=0, use_clones=True, fit_base_estimators=True):
		super().__init__(clfs, voting, weights, verbose, use_clones, fit_base_estimators)
		self.clfs_ = clfs

	def _predict(self, X: np.ndarray) -> np.ndarray:
		"""Collect results from clf.predict calls."""

		if not self.fit_base_estimators:
			return np.asarray([clf(X) for clf in self.clfs_]).T
		else:
			return np.asarray(
				[self.le_.transform(clf(X)) for clf in self.clfs_]
			).T

	def transform(self, X: np.ndarray, dtype: str='int64') -> np.ndarray:
		return self._predict(X).astype(dtype)[0]

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
	df = IMAGE_DF.copy().set_index('Code')
	cols = ["Name_EN", "Element", "Type_EN", "Cost", "Power", "Ex_Burst"]
	mdf = make_database().set_index('Code')[cols]
	df = df.merge(mdf, on="Code", how='left', sort=False)
	df['Ex_Burst'] = df['Ex_Burst'].astype('uint8')

	for category in CATEGORIES:
		model_path = os.path.join(DATA_DIR, "model", f"{category}_model")

		with open(os.path.join(model_path, "category.json")) as fp:
			labels = json.load(fp)

		if category == "Ex_Burst":
			models = [tf.keras.models.load_model(model_path) for model_path in glob.iglob(model_path + "*")]
			voting = MyEnsembleVoteClassifier(models, fit_base_estimators=False) #, weights=[0, 1, 0, 0, 0])
			print(f"{category} transforms")
			print(voting.transform(IMAGES))
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
	df = test_models().reset_index()

	# Remove the ones we know wont work without object detection or full art enablement
	df.drop([2, 3, 4, 17, 26, 32], inplace=True)

	for category in CATEGORIES:
		comp = df[category] == df[f"{category}_yhat"]
		comp = comp.value_counts(normalize=True)

		print(f"{category} accuracy: {comp[True] * 100}%%")
		# print(xf)

	df.sort_index(axis=1, inplace=True)
	print(df)


if __name__ == '__main__':
	main()
