import numpy as np
import pandas as pd
from skimage import io, transform, measure
import glob
import re
import pickle
import time

image_width = 128
image_height = 128

def traindata_loader(path):
    # paths = ['LFW_dataset/match_pairs', 'LFW_dataset/mismatch_pairs']
    # for path in paths:
    #     dirs = os.listdir(path)
    #     for dir in dirs:
    #         images = os.listdir(path+'/'+dir)
    #         for image in images:
    #             print(image)
    # imlist = io.ImageCollection(path)
    # print(imlist[0].shape)
    imlist = glob.glob(path)
    rawImageArray = np.zeros((len(imlist), image_height, image_width), dtype=np.double)
    
    people_names = []
    for i in range(len(imlist)):
        image = imlist[i]
        raw_image = io.imread(image, as_gray=True)
        rawImageArray[i] = transform.resize(raw_image, (image_width, image_height))
        people_name = re.search(r'(?<=\\)[a-zA-Z_-]+', image[28:]) 
        people_names.append(people_name.group(0)[:-1])
    data_label = pd.get_dummies(people_names).to_numpy()
    return rawImageArray, data_label

# convolve with stride=1 and padding=0
# (l, w, h), (fl, fw, h, n), (n) -> (l-fl+1, w-fw+1, n)
def conv(data_in, filter, filter_bias):
    assert len(data_in.shape) == 3 and len(filter.shape) == 4 and len(filter_bias.shape) == 1
    assert data_in.shape[2] == filter.shape[2]    # must share the same height
    assert filter.shape[3] == filter_bias.shape[0]  # each conv kernel has a bias
    input_len, input_width, input_height = data_in.shape
    filter_len, filter_width, filter_height, filter_num = filter.shape
    feature_len = input_len - filter_len + 1
    feature_width = input_width - filter_width + 1
    feature_height = filter_num
    output = np.zeros((feature_len, feature_width, feature_height), dtype=np.double)
    for filt_index in range(feature_height):
        for i in range(feature_len):
            for j in range(feature_width):
                output[i][j][filt_index] = (data_in[i:i+filter_len,j:j+filter_len,:] * filter[:,:,:,filt_index]).sum()
        output[:,:,filt_index] += filter_bias[filt_index]
    return output

# max-feature-map 2/1
# (l, w, h) -> (l, w, h/2)
def mfm(data_in):
    assert len(data_in.shape) == 3
    assert data_in.shape[2]%2 == 0
    input_len, input_width, input_height = data_in.shape
    split = np.zeros((2,input_len, input_width, input_height//2), dtype=np.double)
    split[0] = data_in[:,:,:input_height//2]
    split[1] = data_in[:,:,input_height//2:]
    output = np.amax(split, axis=0)
    assert output.shape == (input_len, input_width, input_height//2)
    return output

# max-pooling with 2*2 filter size and stride=2
# (l, w, h) -> (l/2, w/2, h)
def pool(data_in):
    assert len(data_in.shape) == 3
    assert data_in.shape[0]%2 == 0 and data_in.shape[1]%2 == 0 
    output = measure.block_reduce(data_in, (2,2,1), func=np.max)
    assert output.shape == (data_in.shape[0]//2, data_in.shape[1]//2, data_in.shape[2])
    return output

# pad the first 2 dimension of data_in
# (l, w, h), n -> (n+l+n, n+w+n, h)
def padding(data_in, pad_size):
    assert len(data_in.shape) == 2 or len(data_in.shape) == 3
    assert pad_size > 0
    if len(data_in.shape) == 2:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size] = data_in
    else:
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2, data_in.shape[2]), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size,:] = data_in
    return output

# fully-connected layer
# ndarray, (w, node_num), (node_num) -> (node_num)
def fc(data_in, weights, bias):
    data = data_in.flatten()
    assert data.shape[0] == weights.shape[0]
    assert weights.shape[1] == bias.shape[0]
    weight_num, node_num = weights.shape
    output = np.zeros((node_num), dtype=np.double)
    for i in range(node_num):
        output[i] = (weights[:,i] * data).sum() + bias[i]
    return output

# max-feature-map for fully-connected layer, 2/1
# (node_num) -> (node_num/2)
def mfm_fc(data_in):
    assert len(data_in.shape) == 1
    assert data_in.shape[0]%2 == 0
    node_num = data_in.shape[0]
    split = np.zeros((2, node_num//2), dtype=np.double)
    split[0] = data_in[:node_num//2]
    split[1] = data_in[node_num//2:]
    output = np.amax(split, axis=0)
    print(output.shape)
    assert output.shape == (node_num//2,)
    return output

class LightCNN_9(object):
    def __init__(self, path=None):
        if path != None:
            file = open(path, 'rb')
            data = file.read()
            file.close()
            self.__dict__ = pickle.loads(data)
        else:
            self.conv1_kernel = np.random.randn(5, 5, 1, 96)*np.sqrt(1/(5*5*1))
            self.conv1_bias = np.random.randn(96)
            self.conv2a_kernel = np.random.randn(1, 1, 48, 96)*np.sqrt(1/(1*1*48))
            self.conv2a_bias = np.random.randn(96)
            self.conv2_kernel = np.random.randn(3, 3, 48, 192)*np.sqrt(1/(3*3*48))
            self.conv2_bias = np.random.randn(192)
            self.conv3a_kernel = np.random.randn(1, 1, 96, 192)*np.sqrt(1/(1*1*96))
            self.conv3a_bias = np.random.randn(192)
            self.conv3_kernel = np.random.randn(3, 3, 96, 384)*np.sqrt(1/(3*3*96))
            self.conv3_bias = np.random.randn(384)
            self.conv4a_kernel = np.random.randn(1, 1, 192, 384)*np.sqrt(1/(1*1*192))
            self.conv4a_bias = np.random.randn(384)
            self.conv4_kernel = np.random.randn(3, 3, 192, 256)*np.sqrt(1/(3*3*192))
            self.conv4_bias = np.random.randn(256)
            self.conv5a_kernel = np.random.randn(1, 1, 128, 256)*np.sqrt(1/(1*1*128))
            self.conv5a_bias = np.random.randn(256)
            self.conv5_kernel = np.random.randn(3, 3, 128, 256)*np.sqrt(1/(3*3*128))
            self.conv5_bias = np.random.randn(256)
            self.fc_weights = np.random.randn(8*8*128, 512)
            self.fc_bias = np.random.randn(512)
        return
    def forward(self, data):
        time1 = time.time()
        pad1 = padding(data, 2)
        conv_input = np.zeros((pad1.shape[0], pad1.shape[1], 1), dtype=np.double)
        conv_input[:,:,0] = pad1

        conv1 = conv(conv_input, self.conv1_kernel, self.conv1_bias)
        mfm1 = mfm(conv1)

        pool1 = pool(mfm1)

        conv2a = conv(pool1, self.conv2a_kernel, self.conv2a_bias)
        mfm2a = mfm(conv2a)
        conv2 = conv(padding(mfm2a,1), self.conv2_kernel, self.conv2_bias)
        mfm2 = mfm(conv2)

        pool2 = pool(mfm2)

        conv3a = conv(pool2, self.conv3a_kernel, self.conv3a_bias)
        mfm3a = mfm(conv3a)
        conv3 = conv(padding(mfm3a, 1), self.conv3_kernel, self.conv3_bias)
        mfm3 = mfm(conv3)

        pool3 = pool(mfm3)

        conv4a = conv(pool3, self.conv4a_kernel, self.conv4a_bias)
        mfm4a = mfm(conv4a)
        conv4 = conv(padding(mfm4a, 1), self.conv4_kernel, self.conv4_bias)
        mfm4 = mfm(conv4)

        conv5a = conv(mfm4, self.conv5a_kernel, self.conv5a_bias)
        mfm5a = mfm(conv5a)
        conv5 = conv(padding(mfm5a,1), self.conv5_kernel, self.conv5_bias)
        mfm5 = mfm(conv5)
        
        pool4 = pool(mfm5)

        fc1 = fc(pool4, self.fc_weights, self.fc_bias)
        mfm_fc1 = mfm_fc(fc1)

        time2 = time.time()
        print(time2-time1)
        return mfm_fc1

    def train(self, data, label, epoch, min_batch_size, eta):
        return
    def test(self, data, label):
        return
    def save(self):
        file = open('LightCNN9_model.bin', 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()
        return



if __name__ == "__main__":
    path = 'LFW_dataset/*/*/*.jpg'
    train_data, train_label = traindata_loader(path)
    model = LightCNN_9()
    print(model.forward(train_data[0]))
    