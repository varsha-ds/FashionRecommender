import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import ssl
import pickle

ssl._create_default_https_context = ssl._create_unverified_context

import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#dummy_input = tensorflow.zeros((1, 224, 224, 3))

# Build the model
#model.build(input_shape=dummy_input.shape)

# Print model summary
#print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path, target_size = (224,224))
    img_array = image.img_to_array(img)
    expand_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)
    return normalized_result

#creating a python list to store all the image names
#print(os.listdir('images'))
file_names = []

for file in os.listdir('images'):
    file_names.append(os.path.join('images',file))
 
#we have the path for all the images, now for every file I have to call the function that returns the extracted features of the file.

feature_list = []  #2D list of features, every list has 2048 features, one for each image
for file in file_names:
    feature_list.append(extract_features(file,model))

#print(np.array(feature_list).shape)
    
pickle.dump(feature_list, open('embeddings.pkl','wb'))
pickle.dump(file_names, open('filenames.pkl','wb'))