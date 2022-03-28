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
    """
    Function to load the images stored in the folder named data. The function will retrieve the name of the
    image files and a list with the images as arrays with one channel for each color.
    :param
        folder_path: string, name of the folder where the images are stored
    :returns
        img_names: list of string with the names of the image files."""

    img_names = glob.glob(folder_path+'/*.jpg')

    return img_names


def data_base_histogram(img_names):
    """
    Function to compute the histogram for each of the 3 channel in a set of images. It takes as parameter a list of
    strings with the images names and returns a list with the histogram for each of the image.
    :param
        img_names: list of string with the names of the image files.
    :return:
        img_histograms: list with the histogram for each of the image divided in colors of the RGB.
    """
    img_array = []
    for image in img_names:
        # Load the image from the file in array format
        img_array.append(io.imread(image))
    colors = ("red", "green", "blue")
    channel_ids = (0, 1, 2)
    img_histograms = []
    for image in img_array:
        channel_histograms = []
        for channel_id, c in zip(channel_ids, colors):
            # compute the image histogram for each one of the 3 channels
            histogram, bin_edges = np.histogram(
                image[:, :, channel_id], bins=256, range=(0, 256)
            )
            channel_histograms.append(histogram)

        img_histograms.append(channel_histograms)

    return img_histograms


def compare_histograms(query_image, database_image):
    """
    Function for comparing the histograms and computing the distance between an image and the rest of the database. This
    function will compare the histograms for each of the images.
    :param
        query_image: histogram for each color for the image that is used as query
        database_image: list of histogram for eah of the images in the database
    :return:
        image_distances: list with the distances between the query and each of the images in the database. This list
                         follows the same order of the database.

    """
    img_distances = []
    for image in database_image:
        channel_distances = []
        # Compute the distance for each of the channels
        for img_channel in range(len(image)):
            channel_distances.append(distance.euclidean(image[img_channel], query_image[img_channel]))
        # Add the mean of the distances divided by the number of pixels to the list of distances
        img_distances.append(np.mean(channel_distances)/sum(query_image[img_channel]))

    return img_distances


def create_kernels(n_theta=4, sigmas=(1, 3), frequencies=(0.05, 0.25)):
    """
    This function will generate the different kernels that are used for computing the gabor filter.
    :param
        n_theta: integer, number of orientation for the filter (the orientation will be computed dividing the circle in
                 the number of thetas.
        sigmas: list of integers. Parameter for the g distribution to be use for generating the kernel.
        frequencies: list of floats. Parameter for the g distribution to be use for generating the kernel.
    :return:
        kernels: list of kernels to be used for computing the the gabor filter
    """
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
    """
    Function use fo compute the feature vector of textures for each image.
    :param
        img_names: List of string, names of the files that contains each image
        kernels: list of kernels to be use for computing the gabor filter.
    :return:
        img_feats: list of vectors. Each vector is the feature vector for each of the image obtained after the gabor
                   filter.
    """
    img_feats = []
    for image in img_names:
        # Load the images as grayscale for the filter
        image = io.imread(image, as_gray=True)
        feats = []
        # Perform the convolution between the image and each one of the kernels selected.
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            # Compute the mean and standard of the image to generate the feature vector of the image
            feats.append(filtered.mean())
            feats.append(filtered.var())
        img_feats.append(feats)
    return img_feats


def compare_gabor(gabor_query, gabor_data):
    """
    Function for computing the distance between the query image and the different images in the database with the gabor
    filter for the texture.
    :param
        gabor_query: list of floats, feature vector for the query image.
        gabor_data:  list of list of floats, each one represent the feature vector for each of the images presented in
                     the database.
    :return:
        gabor_distance_scaled: list of floats. Result of the canberra distance between each of the images and the query,
                               this distance is scaled with a min max scaler.
    """
    gabor_dist = []
    for data in gabor_data:
        gabor_dist.append(distance.canberra(data, gabor_query))

    gabor_dist_scaled = MinMaxScaler().fit_transform(np.array(gabor_dist).reshape(-1, 1))

    return gabor_dist_scaled


def sort_distances(dist_data, img_names):
    """
    Function that sort the images names with the distance information.

    :param
        dist_data: list of distances for the image with a singular method.
        img_names: list of string with the names for each image file.
    :return:
        complete_results: list of tuples (image file name, and punctuation respect to the query image), ordered from the
                          most similar to the less.
    """
    complete_result = sorted(zip(img_names, dist_data), key=lambda x: x[1])

    return complete_result


'''images_names = load_images("data")

images_histograms = data_base_histogram(images_names)
histogram_distances = compare_histograms(images_histograms[20], images_histograms)
print(histogram_distances[:5])

gabor_kernels = create_kernels()
gabor_feats = compute_feats(images_names, gabor_kernels)
gabor_distances = compare_gabor(gabor_feats[20], gabor_feats)
print(gabor_distances[:5])

similar_images_hist = sort_distances(histogram_distances, images_names)
similar_images_gabor = sort_distances(gabor_distances, images_names)
ranked_images_names, _ = zip(*similar_images_gabor)

ranked_images_histograms = data_base_histogram(ranked_images_names[:10])
histogram_distances_ranked = compare_histograms(ranked_images_histograms[0], ranked_images_histograms)
similar_images_hist_gabor = sort_distances(histogram_distances_ranked, ranked_images_names[:10])

img = mpimg.imread(similar_images_hist_gabor[0][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images_hist_gabor[1][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images_hist_gabor[5][0])
plt.imshow(img)
plt.show()
img = mpimg.imread(similar_images_hist_gabor[-1][0])
plt.imshow(img)
plt.show()
'''
