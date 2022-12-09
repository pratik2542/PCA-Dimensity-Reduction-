# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:09:12 2022

@author: Pratik
"""

from sklearn.datasets import fetch_openml
import pandas as pd

mnist = fetch_openml('mnist_784')
print(mnist.data.shape)

from sklearn.model_selection import train_test_split
X = mnist ['data']
Y = mnist ['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


print(Y_train.shape)

y = pd.DataFrame(Y_train).to_numpy()



print(X_train.shape)

import matplotlib.pyplot as plt
import matplotlib as mpl


#printing numbers
for i in range(30):
    some_digit1=X_train.iloc[i]
    #Plot the image using imshow
    some_digit_image1 = some_digit1.values.reshape((28, 28))
    plt.imshow(some_digit_image1, cmap=mpl.cm.binary)
    plt.show()

#scaling
#standardization
from sklearn.preprocessing import StandardScaler
# define data
print(X_train)
# define standard scaler
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(X_train)
print(scaled)

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
#X_train_reduced = pca.fit_transform(X_train)
pca.fit(scaled)
x_pca=pca.transform(scaled)

x_pca

scaled.shape

#shape PCA
x_pca.shape

pca.explained_variance_ratio_

#2D
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=y)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


#incremental PCA
from sklearn.decomposition import IncrementalPCA
import numpy as np

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") 
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)


X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)


# EXTRA
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row):
    n_rows = (len(instances) - 1) // images_per_row + 1

    # Append empty images to fill the end of the grid, if needed:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # Reshape the array so it's organized as a grid containing 28×28 images:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis),
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we
    # want to combine next to each other, using transpose(), and only then we
    # can reshape:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # Now that we have a big image, we just need to show it:
    plt.imshow(big_image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.subplot(122)
plot_digits(X_recovered_inc_pca[::2100])
plt.tight_layout()
