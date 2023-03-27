<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[![Logo][logo]][logo-url]

# CrystalVision

This is a machine learning project aimed at classifying Final Fantasy Trading Card Game cards based on their attributes -- such as card name, element, rarity, and abilities. The goal is to create a model that can accurately predict the card's unique identifier ("Code") given its features.

## Dataset

The dataset used for this project is a collection of Final Fantasy Trading Card Game cards obtained from [Square Enix's official website](https://fftcg.square-enix-games.com/na). The dataset includes card name, element, rarity, type, power, and abilities.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Approach

The approach used for this project is supervised learning, specifically classification. The dataset is split into training and testing sets. The model is trained on the training set and evaluated on the testing set. Once a model has finished being fit and saved, it is then tested against real-world hand-pick images from the internet that have unaccounted attributes such as, it is foil, in a different language, or has some border.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Dependencies

![python-shield]

This project is implemented using Python 3.X. 

![tf-shield]
![keras-shield]
![np-shield]
![pd-shield]
![skl-shield]
![sci-shield]

All libraries used are included in the `requirements.txt`. You can execute the following to update your dependencies

```
$ pip install -r requirements.txt
```


Moreover, the sequential model were originally created with the Nvidia GeForce GTX 980 GPU and Intel Core i7-5960X CPU. Therefore the complexity of the models is further constrained by allowable training time and gpu memory.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

To run this project, simply execute the following commands:

```
python .\src\gatheredata.py
```

This will gather all the required training/testing images and created a `.\data\cards.json` file with all relevent card data.

```
python .\src\generatemodels.py
```

This will train the model. You can modify the code to change the classification algorithm used, or to include additional features.

```
python .\src\testmodels.py
```

This will test the models against real-world hand-picked data and evaluate our accuracy. Images will be cached (downloaded) and converted to JPEG (removing any alpha channels) into `.\data\test\` 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Conclusion

This project demonstrates the feasibility of using machine learning to classify Final Fantasy Trading Card Game cards based on their attributes. Future work could involve expanding the dataset to include more cards and features, exploring other classification algorithms, and (multi)object detection.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Acbarakat/CrystalVision.svg?style=for-the-badge
[contributors-url]: https://github.com/Acbarakat/CrystalVision/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Acbarakat/CrystalVision.svg?style=for-the-badge
[forks-url]: https://github.com/Acbarakat/CrystalVision/network/members
[stars-shield]: https://img.shields.io/github/stars/Acbarakat/CrystalVision.svg?style=for-the-badge
[stars-url]: https://github.com/Acbarakat/CrystalVision/stargazers
[issues-shield]: https://img.shields.io/github/issues/Acbarakat/CrystalVision.svg?style=for-the-badge
[issues-url]: https://github.com/Acbarakat/CrystalVision/issues
[license-shield]: https://img.shields.io/github/license/Acbarakat/CrystalVision.svg?style=for-the-badge
[license-url]: https://github.com/Acbarakat/CrystalVision/blob/main/LICENSE.md
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/allonte-barakat/
[python-shield]: https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue
[tf-shield]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[keras-shield]: https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white
[np-shield]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[pd-shield]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[skl-shield]: https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sci-shield]: https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white
[logo]: https://repository-images.githubusercontent.com/616501925/3051c914-b18a-42ab-96e1-96d6fb7e2b81
[logo-url]: https://github.com/Acbarakat/CrystalVision
