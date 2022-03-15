# This is a sample Python script.
import glob
from skimage import io
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from sklearn.preprocessing import MinMaxScaler


def load_images(folder_path):
    img_names = glob.glob(folder_path+'/*.jpg')
    img_array = []
    for image in img_names:
        img_array.append(io.imread(image))
    return img_names, img_array


def data_base_histogram(img_array):
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    img_histograms = []
    for image in img_array:
        channel_histograms = []
        for channel_id, c in zip(channel_ids, colors):
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            channel_histograms.append(histogram)

        img_histograms.append(channel_histograms)

    return img_histograms


def compare_histograms(query_image, database_image, img_names):
    img_distances = []
    for image in database_image:
        channel_distances = []
        for img_channel in range(len(image)):
            channel_distances.append(distance.euclidean(image[img_channel], query_image[img_channel]))
        img_distances.append(np.mean(channel_distances)/sum(query_image[img_channel]))

    img_distances_scaled = MinMaxScaler().fit_transform(np.array(img_distances).reshape(-1, 1))

    return img_distances_scaled


def create_kernels(n_theta=4, sigmas=(1, 3), frequencies=(0.05, 0.25)):
    kernels = []
    for theta in range(n_theta):
        theta = theta / 4. * np.pi
        for sigma in sigmas:
            for frequency in frequencies:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def compute_feats(img_names, kernels):
    img_feats = []
    for image in img_names:
        image = io.imread(image, as_gray=True)
        feats = []
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats.append(filtered.mean())
            feats.append(filtered.var())
        img_feats.append(feats)
    return img_feats


def compare_gabor(gabor_query, gabor_data, img_names):
    gabor_dist = []
    for data in gabor_data:
        gabor_dist.append(distance.canberra(data, gabor_query))

    gabor_dist_scaled = MinMaxScaler().fit_transform(np.array(gabor_dist).reshape(-1, 1))

    return gabor_dist_scaled


def combine_distances(hist_dist, gabor_dist, img_names):
    result = np.add(hist_dist, gabor_dist)
    complete_result = sorted(zip(img_names, result), key=lambda x: x[1])

    return complete_result


images_names, images_array = load_images("data")
images_histograms = data_base_histogram(images_array)


histogram_distances = compare_histograms(images_histograms[20], images_histograms, images_names)
print(histogram_distances[:5])

gabor_kernels = create_kernels()
gabor_feats = compute_feats(images_names, gabor_kernels)
gabor_distances = compare_gabor(gabor_feats[20], gabor_feats, images_names)
print(gabor_distances[:5])

similar_images = combine_distances(histogram_distances, gabor_distances, images_names)
print(similar_images[:5])
print(similar_images[-5:])

img = mpimg.imread(similar_images[0][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images[1][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images[11][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images[-1][0])
plt.imshow(img)
plt.show()
