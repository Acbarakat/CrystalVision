# -*- coding: utf-8 -*-
"""
Generate various models

Todo:
    * Find more varied data, such as off-center or card with background

"""
import os
import json
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, applications, models
from sklearn.model_selection import train_test_split
from keras.utils import image_utils
from keras.utils.image_dataset import paths_and_labels_to_dataset

from gatherdata import CARD_API_FILEPATH, DATA_DIR


def make_database() -> pd.DataFrame:
    '''
    Load card data and clean up any issue found in the API

    Returns:
        Card API dataframe
    '''
    with open(CARD_API_FILEPATH) as fp:
        data = json.load(fp)["cards"]

    df = pd.DataFrame(data)
    df["thumbs"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["thumbs"]])
    df["images"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["full"]])
    df["Ex_Burst"] = df["Ex_Burst"].apply(lambda i: i == "\u25cb").astype(bool)
    df["Element"] = df["Element"].str.replace("/", "_")
    df["Power"] = df['Power'].str.replace(" ", "").replace("\u2015", "").replace("\uff0d", "")

    # Ignore Crystal Tokens
    df = df.query("Type_EN != 'Crystal'")

    # Ignore Boss Deck cards
    df = df.query("Rarity != 'B'")  

    return df


def make_model(train_ds: tf.data.Dataset,
               validation_ds: tf.data.Dataset,
               image_shape: tuple,
               label_count: int,
               model_type: str="resnet",
               epochs: int=30,
               seed: typing.Any=None,
               model_name: str="fftcg_model") -> None:
    '''
    Create and save model

    Args:
        train_ds (tf.data.Dataset): Training data
        validation_ds (tf.data.Dataset): Test data
        image_shape (tuple): The (height, width) of the image data
        label_count (int): The number of `labels`
        model_type (str): String describing the type of `model`
            (default is `resnet`)
        epochs (int): The maximum number of training iterations
            (default is `30`)
        seed (Any): Optional random seed for shuffling, transformations,
            and dropouts.
        model_name (str): The name of the model
            (default is `fftcg_model`)
    '''
    tf.keras.backend.clear_session()

    if model_type == "custom":
        # TODO: A well crafted custom per attribute is required
        model = models.Sequential(name="custom", layers=[
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            layers.Dense(2 ** 6, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.RMSprop()
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True)
    elif model_type == "cost":
        model = models.Sequential(name="cost", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 5, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        optimizer = tf.keras.optimizers.RMSprop(centered=True)
    elif model_type == "power":
        model = models.Sequential(name="power", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Dense(2 ** 6, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.RMSprop()
    elif model_type == "element":
        model = models.Sequential(name="element", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.AveragePooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(2 ** 6, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.RMSprop()
    elif model_type == "type_en":
        model = models.Sequential(name="type_en", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    elif model_type == "resnet":
        model = applications.ResNet50V2(weights=None, input_shape=image_shape, classes=label_count)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
    else:
        raise RuntimeError(f"model_type '{model_type}' is not supported")

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    # print(model.summary())
    print(model.name)

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        # tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', min_delta=0.0010, patience=3),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                         min_delta=0.0025,
                                         patience=4,
                                         restore_best_weights=True),
    ]

    model.fit(train_ds,
              epochs=epochs,
              validation_data=validation_ds,
              callbacks=callbacks)
    
    model.save(os.path.join(DATA_DIR, "model", model_name))


def generate(df: pd.DataFrame,
             key: str,
             image_key: str,
             stratify: list,
             model_type: str="resnet",
             image_size: tuple=(250, 179),
             label_mode: str='categorical',
             batch_size: int=32,
             shuffle: bool=True,
             seed: typing.Any=None,
             interpolation: str="bilinear",
             crop_to_aspect_ratio: bool=False) -> None:
    '''
    Create and save model and catergory data

    Args:
        df (pd.DataFrame): All card information
        key (str): The column name of image sources
        image_key (str): The (height, width) of the image data
        stratify (list): The columns by which we stratify training and test datasets
        model_type (str): String describing the type of `model`
            (default is `resnet`)
        image_shape (tuple): The (height, width) of the image data
            (default is (250, 179))
        label_mode (str): String describing the encoding of `labels`. Options are:
          - 'int': means that the labels are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the labels are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
          - 'binary' means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0 or 1
              (e.g. for `binary_crossentropy`).
          - None (no labels).
        batch_size (int): Size of the batches of data.
            If `None`, the data will not be batched
            (default is 32)
        shuffle (bool): Whether to shuffle the data.
            If set to False, sorts the data in alphanumeric order.
            (default is True)
        seed (Any): Optional random seed for shuffling and transformations.
        interpolation (str): The interpolation method used when resizing images.
            Supports `bilinear`, `nearest`, `bicubic`, `area`,
            `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
            (default is `bilinear`)
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the target
            aspect ratio, the output image will be cropped so as to return the
            largest possible window in the image (of size `image_size`) that matches
            the target aspect ratio.
            (default is False)
    '''
    codes, uniques = df[key].factorize()

    X_train, X_test, y_train, y_test = train_test_split(df[image_key],
                                                        codes,
                                                        test_size=0.33,
                                                        random_state=seed,
                                                        stratify=df[stratify])

    if model_type == "resnet":
        preprocess_layer = applications.resnet_v2.preprocess_input
    else:
        preprocess_layer = layers.Rescaling(1. / 255)

    training_dataset = paths_and_labels_to_dataset(
        image_paths=X_train.tolist(),
        image_size=image_size,
        num_channels=3,
        labels=y_train.tolist(),
        label_mode=label_mode,
        num_classes=len(uniques),
        interpolation=interpolation,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            training_dataset = training_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        training_dataset = training_dataset.batch(batch_size)
    else:
        if shuffle:
            training_dataset = training_dataset.shuffle(buffer_size=1024, seed=seed)
    
    training_dataset = training_dataset.map(tf.autograph.experimental.do_not_convert(lambda x, y: (preprocess_layer(x), y)))
    flipped_training_dataset= training_dataset.map(lambda x, y: (tf.image.flip_up_down(x), y))
    training_dataset = training_dataset.concatenate(flipped_training_dataset)

    training_dataset.class_names = uniques.tolist()
    training_dataset.file_paths = X_train.tolist()
    training_dataset.element_spec[0]._name = 'train'
    # print(training_dataset)

    testing_dataset = paths_and_labels_to_dataset(
        image_paths=X_test.tolist(),
        image_size=image_size,
        num_channels=3,
        labels=y_test.tolist(),
        label_mode=label_mode,
        num_classes=len(uniques),
        interpolation=interpolation,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )
    testing_dataset = testing_dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            testing_dataset = testing_dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        testing_dataset = testing_dataset.batch(batch_size)
    else:
        if shuffle:
            testing_dataset = testing_dataset.shuffle(buffer_size=1024, seed=seed)
    testing_dataset = testing_dataset.map(lambda x, y: (preprocess_layer(x), y))
    flipped_testing_dataset = testing_dataset.map(lambda x, y: (tf.image.flip_up_down(x), y))
    testing_dataset = testing_dataset.concatenate(flipped_testing_dataset)

    testing_dataset.class_names = uniques.tolist()
    testing_dataset.file_paths = X_test.tolist()
    testing_dataset.element_spec[0]._name = 'test'
    # print(testing_dataset)

    KEY_MODEL_PATH = os.path.join(DATA_DIR, "model", f"{key}_model")
    if not os.path.exists(KEY_MODEL_PATH + os.sep):
        os.makedirs(KEY_MODEL_PATH + os.sep)

    with open(os.path.join(KEY_MODEL_PATH, "category.json"), "w+") as fp:
        json.dump(uniques.to_list(), fp)

    make_model(training_dataset,
               testing_dataset,
               seed=seed,
               model_type=model_type,
               model_name=f"{key}_model",
               image_shape=(image_size[0], image_size[1], 3),
               label_count=len(uniques))


def main(image: str="thumbs",
         seed: typing.Any=None,
         interpolation: str="bilinear",
         default_model_type: str="resnet") -> None:
    '''
    Create and save all models based on card data

    Args:
        image (str): The column to explode containing lists of image sources.
            (default is `thumbs`)
        seed (Any): Optional random seed for shuffling and transformations.
            (default is None)
        interpolation (str): The interpolation method used when resizing images.
            Supports `bilinear`, `nearest`, `bicubic`, `area`,
            `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
            (default is `bilinear`)
        default_model_type (str): String describing the type of `model`. Options are:
          - 'custom': Custom model, using RMSprop optimizer, and 1/255 scale preprocessor.
          - 'resnet' ResNet50V2, using SGD optimizer, and ResnetV2 preprocessor.
          (default is `resnet`)
    '''
    if seed is None:
        seed = np.random.randint(1e6)
    interpolation = image_utils.get_interpolation(interpolation)

    df = make_database().explode(image)
    # df["Class"] = df['thumbs'].apply(lambda x: 'Full Art' if '_fl' in x.lower() else '')

    # Ignore Full Art Cards
    df = df.query(f"~{image}.str.contains('_FL') and ~{image}.str.contains('_2_')")

    # Ignore Promo Cards
    df = df.query(f"~{image}.str.contains('_PR')")

    # Ignore
    df = df.query(f"~{image}.str.contains('_premium')")

    # Ignore multi-element cards
    df = df.query("~Element.str.contains('_')")

    # Ignore by langyage
    # df = df.query(f"~{image}.str.contains('_eg')")  # English
    df = df.query(f"~{image}.str.contains('_fr')")  # French
    # df = df.query(f"~{image}.str.contains('_es')")  # Spanish
    df = df.query(f"~{image}.str.contains('_it')")  # Italian
    df = df.query(f"~{image}.str.contains('_de')")  # German
    df = df.query(f"~{image}.str.contains('_jp')")  # Japanese

    # WA: Bad Download/Image from server
    df = df.query(f"{image} not in ('8-080C_es.jpg', '11-138S_fr.jpg', '12-049H_fr_Premium.jpg', '13-106H_de.jpg')")

    # Source image folder
    df = df.copy()  # WA: for pandas modification on slice
    if image == "images":
        df[image] = os.path.abspath(os.path.join(DATA_DIR, "img")) + os.sep + df[image]
    else:
        df[image] = os.path.abspath(os.path.join(DATA_DIR, "thumb")) + os.sep + df[image]

    model_mapping = (
        # ("Name_EN", ["Name_EN", "Element", "Type_EN"], "resnet"),
        ("Element", ["Element", "Type_EN"], "element"),
        ("Type_EN", ["Type_EN", "Element"], "type_en"),
        ("Cost", ["Cost", "Element"], "cost"),
        ("Power", ["Power", "Type_EN", "Element"], "power"),
    )

    for key, stratify, model_type in model_mapping:
        generate(df,
                 key,
                 image,
                 stratify,
                 model_type,
                 seed=seed,
                 interpolation=interpolation)


if __name__ == '__main__':
    with tf.device('/GPU:0'):
        main(seed=23)
