import numpy as np
from skimage import measure

def conv(data_in, filter, filter_bias):
    '''
      Convolve with stride=1 and padding=0
      (l, w, h), (fl, fw, h, n), (n) -> (l-fl+1, w-fw+1, n)
    '''
    assert len(data_in.shape) == 3 and len(filter.shape) == 4 and len(filter_bias.shape) == 1
    assert data_in.shape[2] == filter.shape[2]    # must share the same height
    assert filter.shape[3] == filter_bias.shape[0]  # each conv kernel has a bias
    input_len, input_width, input_height = data_in.shape
    filter_len, filter_width, filter_height, filter_num = filter.shape
    feature_len = input_len - filter_len + 1
    feature_width = input_width - filter_width + 1
    feature_height = filter_num
    img2col = np.lib.stride_tricks.as_strided(data_in, shape=(feature_len, feature_width, filter_len, filter_width, filter_height), strides=(8*input_height*input_width, 8*input_height, 8*input_height*input_width, 8*input_height, 8))
    img2col = np.reshape(img2col, (feature_len, feature_width, filter_len*filter_width*filter_height))
    filter = np.reshape(filter, (filter_len*filter_width*filter_height, filter_num))
    output = np.add(np.matmul(img2col, filter), filter_bias)
    assert output.shape == (feature_len, feature_width, feature_height)
    return output

def mfm(data_in):
    '''
    ## max-feature-map 2/1
    (l, w, h) -> (l, w, h/2)  
    '''
    assert len(data_in.shape) == 3
    assert data_in.shape[2]%2 == 0
    input_len, input_width, input_height = data_in.shape
    split = np.zeros((2,input_len, input_width, input_height//2), dtype=np.double)
    split[0] = data_in[:,:,:input_height//2]
    split[1] = data_in[:,:,input_height//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:,:,:input_height//2] = repmax[:,:,input_height//2:] = output
    location = (repmax == data_in)
    assert output.shape == (input_len, input_width, input_height//2)
    assert location.shape == data_in.shape
    return output, location

def pool(data_in):
    '''
    ## max-pooling with 2*2 filter size and stride=2
    (l, w, h) -> (l/2, w/2, h)
    '''
    assert len(data_in.shape) == 3
    assert data_in.shape[0]%2 == 0 and data_in.shape[1]%2 == 0 
    output = measure.block_reduce(data_in, (2,2,1), func=np.max)
    repmax = np.repeat(np.repeat(output, 2, axis=0), 2, axis=1)
    location = (repmax == data_in)
    assert output.shape == (data_in.shape[0]//2, data_in.shape[1]//2, data_in.shape[2])
    assert location.shape == data_in.shape
    return output, location

def padding(data_in, pad_size):
    '''
    ## pad the first 2 dimension of data_in
     (l, w, h), n -> (n+l+n, n+w+n, h)
    '''
    assert len(data_in.shape) == 2 or len(data_in.shape) == 3
    assert pad_size > 0
    if len(data_in.shape) == 2: # 2D padding
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size] = data_in
    else: # 3D padiing
        output = np.zeros((data_in.shape[0]+pad_size*2, data_in.shape[1]+pad_size*2, data_in.shape[2]), dtype=np.double)
        output[pad_size:-pad_size,pad_size:-pad_size,:] = data_in
    return output

def fc(data_in, weights, bias):
    '''
    ## fully-connected layer
    ndarray, (w, node_num), (node_num) -> (node_num)
    '''
    data = data_in.flatten()
    assert data.shape[0] == weights.shape[0]
    assert weights.shape[1] == bias.shape[0]
    weight_num, node_num = weights.shape
    output = np.matmul(data, weights) + bias
    assert output.shape == (node_num,)
    return output

def mfm_fc(data_in):
    '''
    ## max-feature-map for fully-connected layer, 2/1
    (node_num) -> (node_num/2)
    '''
    assert len(data_in.shape) == 1
    assert data_in.shape[0]%2 == 0
    node_num = data_in.shape[0]
    split = np.zeros((2, node_num//2), dtype=np.double)
    split[0] = data_in[:node_num//2]
    split[1] = data_in[node_num//2:]
    output = np.amax(split, axis=0)
    repmax = np.zeros(data_in.shape, dtype=np.double)
    repmax[:node_num//2] = repmax[node_num//2:] = output
    location = (repmax == data_in)
    assert output.shape == (node_num//2,)
    assert location.shape == data_in.shape
    return output, location

def softmax(data_in):
    '''
    ## softmax layer
    (3095) -> (3095)
    '''
    m = np.amax(data_in)
    data_in -=m
    e = np.exp(data_in)
    s = np.sum(e)
    output = e/s
    return output 

def cross_entropy(data_in, label_vec):
    '''
    ## cross entropy as loss function. 
    (3095) -> 1
    '''
    l = np.log(data_in)
    return -np.dot(l, label_vec)