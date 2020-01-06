import numpy as np
import pandas as pd
from skimage import io, transform
import glob
import re
image_width = 128
image_height = 128

def traindata_loader(path):
    '''
    Load data and label from train dataset
    '''
    imlist = glob.glob(path)
    rawImageArray = np.zeros((len(imlist), image_height, image_width), dtype=np.double)
    people_names = []
    for i in range(len(imlist)):
        image = imlist[i]
        raw_image = io.imread(image, as_gray=True)
        # Preprocessing
        rawImageArray[i] = transform.resize(raw_image[25:-25][25:-25], (image_width, image_height))
        rawImageArray[i] = (rawImageArray[i]-np.amin(rawImageArray[i]))/(np.amax(rawImageArray[i])-np.amin(rawImageArray[i]))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[20:]) # Extract people's name
        people_names.append(people_name.group(0)[:-1])  
    
    # One-hot encode
    data_label = pd.get_dummies(people_names).to_numpy() 
    data_label_expand = np.zeros((data_label.shape[0], 3095))
    data_label_expand[:, :data_label.shape[1]] = data_label
    return rawImageArray, data_label_expand