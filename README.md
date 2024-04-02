# FashionRecommender

CNN - Resnet (trained on Images)/ customized CNN model building
using keras
Transfer Learning


Importing the model(Resnet module)
Extract features (from the Images)
Export Features
Generate Recommendations

CNN- has multiple layers, each layer identifies different features. Instead of analysing each and every pixel, it analyzes the different features at different layers.

Resnet model extracts the features from the images

For every image, Resnet model extracts 2048 features
i.e., [44k(total number of images),2048(number of features)]

Based on the features, we can find the similar and different images.

Image1[1,2,....2048]
Image2[1,2,....2048]
...
Image44k[1,2,....2048]

It means every we have 44k vectors with 2048 dimensionality
All these vectors are present in the 3D/2048D Vector space.
If I give a new Image, Resnet generates a new vector with 2048 dimensions.
The logic behind generating the recommendations is based on the location of the new vector, what are the closest vectors present in the vector space will be considered as similar and are the closest recommendations.
To Summarize, our model calculates the euclidean distance between the new vector and the existing 44 thousand vectors, the nearest 5 vectors will be considered as the neighbours and the respective recommendations are shown.


Embeddings- Representing every image in 2048dimensions


import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())