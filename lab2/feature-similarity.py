from convnets import *
import os
import random
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn
%matplotlib osx
from PIL import Image
#3361344.jpg

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="alexnet_weights.h5", heatmap=True)
model.compile(optimizer=sgd, loss='mse')

model.layers.pop()
model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output]
image_path = '/Users/alexis/Documents/upf-ml-recsys/lab2/images'

#files = random.sample(os.listdir(image_path),100)
# get index [i for i, elem in enumerate(files) if '361344.jpg' in elem]

files = os.listdir(image_path)
img = preprocess_image_batch([os.path.join(image_path, file) for file in files])

image_features = model.predict(img)

cosine_similarity = [1.0 - spatial.distance.cosine(image_features[1,],image_features[i,]) for i in xrange(len(files))]
cosine_indexes = np.argsort(cosine_similarity)

f, axes = plt.subplots(1, 1, figsize=(2,2))
axes.imshow((Image.open(os.path.join(image_path,files[1]))))
axes.axis('off')

sim_files = [files[i] for i in cosine_indexes]

bottom5 = sim_files[0:5]

top5 = sim_files[-5:]


f, axes = plt.subplots(1, 5, figsize=(15,2))


for j in reversed(xrange(5)):
    axes[4 - j].imshow(Image.open(os.path.join(image_path,bottom5[j])))
    axes[4 - j].axis('off')

f, axes = plt.subplots(1, 5, figsize=(15,2))

for j in reversed(xrange(5)):
    axes[4 - j].imshow(Image.open(os.path.join(image_path,top5[j])))
    axes[4 - j].axis('off')
