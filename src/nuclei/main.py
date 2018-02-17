import pandas as pd
import matplotlib.image as mpimg
import cv2
import numpy as np
import glob

from skimage.filters import threshold_otsu
from scipy import ndimage

from nuclei.utils import rle_encoding
from constants import ROOT


def analyze_image(img_path):
    '''
    Take an image_path, preprocess and label it, extract the RLE strings
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale

    img_id = img_path.split('/')[-3]
    img = mpimg.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(img_gray)
    mask = np.where(img_gray > thresh_val, 1, 0)
    if np.sum(mask == 0) < np.sum(mask == 1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)

    # Loop through labels and add each to a DataFrame
    img_df = pd.DataFrame()
    for label_num in range(1, nlabels + 1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': img_id, 'EncodedPixels': rle})
            img_df = img_df.append(s, ignore_index=True)

    return img_df


def analyze_list_of_images(img_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for img_path in img_path_list:
        img_df = analyze_image(img_path)
        all_df = all_df.append(img_df, ignore_index=True)

    return all_df


if __name__ == '__main__':
    DATA_DIR = ROOT + '/data/'
    test_files = glob.glob(DATA_DIR + 'stage1_test/*/images/*.png')
    df = analyze_list_of_images(test_files)
    df.to_csv('submission.csv', index=None)

