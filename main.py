# This is a sample Python script.
import glob
import skimage
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def data_base_histogram(folder_path):
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    img_names = glob.glob(folder_path+'/*.jpg')
    images_array = []
    for img in img_names:
        images_array.append(skimage.io.imread(img))

    img_histograms = []
    for image in images_array:
        channel_histograms = []
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            channel_histograms.append(histogram)

        img_histograms.append(channel_histograms)

    return img_names, img_histograms


def compare_histograms(query_image, database_image, img_names):
    img_distances = []
    for img in database_image:
        channel_distances = []
        for img_channel in range(len(img)):
            channel_distances.append(distance.euclidean(img[img_channel], query_image[img_channel]))
        img_distances.append(np.mean(channel_distances)/sum(query_image[img_channel]))
    result = sorted(zip(img_names, img_distances), key=lambda x: x[1])

    return result


images_names, images_histograms = data_base_histogram("data")
distances = compare_histograms(images_histograms[20], images_histograms, images_names)
print(distances[:5])

img = mpimg.imread(distances[0][0])
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread(distances[1][0])
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread(distances[10][0])
imgplot = plt.imshow(img)
plt.show()
img = mpimg.imread(distances[-1][0])
imgplot = plt.imshow(img)
plt.show()



