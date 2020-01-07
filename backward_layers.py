import numpy as np
from forward_layers import conv, padding
def get_derivative_softmax(fc2_output, label_vec):
    '''
    fc2_output: (3095,1)
    return (3095,1)
    '''
    return fc2_output - label_vec

def get_derivative_fcout(input_vec, fc_weights, fc_bias, bp_gradient):
    '''
    y = Wx + b
    bp_gradient: 从后续layer传来的梯度 (n, 1)
    fc2: input_vec: (256,) bp_gradient: (3095,), fc_weights:(256,3095)
    '''
    dw = np.matmul(input_vec[:,None],bp_gradient[:,None].transpose())
    assert dw.shape == fc_weights.shape
    db = bp_gradient
    db = db[:,None]
    assert db.shape == fc_bias[:,None].shape
    dx = np.matmul(fc_weights, bp_gradient[:,None])
    return dw,db,dx

def get_derivate_mfm_fc1(location, bp_gradient):
    '''
    input_vec: (512,)
    bp_gradient: (256,1)
    '''
    tmp = np.vstack((bp_gradient,bp_gradient))
    assert tmp.shape[0] == 512
    output = tmp * location[:,None].astype(int) 
    assert output.shape == (512,1)
    return output

def get_derivative_fc(input_vec, fc_weights, fc_bias, bp_gradient):
    '''
    y = Wx + b
    bp_gradient: 从后续layer传来的梯度 (n, 1)
    fc1: input_vec: (8*8*128,) bp_gradient: (512,1), fc_weights:(8*8*128,512)
    '''
    dw = np.matmul(input_vec[:,None],bp_gradient.transpose())
    assert dw.shape == fc_weights.shape
    db = bp_gradient
    assert db.shape == fc_bias[:,None].shape
    dx = np.matmul(fc_weights, bp_gradient)
    assert dx.shape == (8*8*128,1)
    return dw,db,dx
    
def rot180(conv_filters):
    '''
    将卷积核旋转180度
    conv_filters: (fl, fw, h, n) -> (fl, fw, h, n)
    '''
    rot180_filters = np.flip(np.flip(conv_filters, 0), 1)
    return rot180_filters

def get_derivative_conv(input_img, filter, filter_bias, bp_gradient, conv_output):
    '''
    对w求导的原理是：将bp_gradient的每一层[:,:,i]作为新的filter 对input_img的每一层[:,:,j]进行卷积
    input_img: the input of convolution layer
    filter, filter_bias: the w and b of current convolution layer 
    bp_gradient: 从后续层传来的梯度
    conv_output: the output feature map of current convolution layer 
    '''
    # 记录必要的参数
    filter_size = filter.shape[0]
    input_img_size= input_img.shape[0]
    input_img_channal = input_img.shape[2] 
    output_channal = conv_output.shape[-1]

    dw = np.zeros(filter.shape)            
    bp_gradient_filter_size = bp_gradient.shape[0]
    feature_size = input_img_size - bp_gradient_filter_size + 1

    # img2col技术，将要做卷积的区域reshape成一个向量
    input_img2col = np.lib.stride_tricks.as_strided(input_img, shape=(feature_size, feature_size, input_img_channal, bp_gradient_filter_size, bp_gradient_filter_size), strides=(8*input_img_channal*input_img_size, 8*input_img_channal, 8, 8*input_img_channal*input_img_size, 8*input_img_channal))
    input_img2col = np.reshape(input_img2col, (feature_size, feature_size, input_img_channal,bp_gradient_filter_size*bp_gradient_filter_size))
    # 对filter reshape, 将其reshape成一个向量
    bp_gradient_filter = bp_gradient.reshape(( bp_gradient_filter_size* bp_gradient_filter_size, bp_gradient.shape[-1]))
    # 将卷积运算转化为矩阵相差
    dw = np.matmul(input_img2col,bp_gradient_filter)
    
    assert dw.shape == filter.shape

    tmp_bias = np.zeros(input_img.shape[-1])
    # 对filter做180度翻转
    rot_filter = rot180(filter).swapaxes(-2,-1)
    if filter.shape[0]!= 1:
        dx = conv(padding(bp_gradient,filter.shape[0]-1), rot_filter, tmp_bias)[1:-1,1:-1,:]
    else:
        dx = conv(bp_gradient, rot_filter, tmp_bias)

    db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
    assert db.shape == filter_bias.shape
    return dw,db,dx      

def get_derivative_conv2pool(input_img, filter, filter_bias, bp_gradient):
    '''
    针对pool的输出跳过一层卷积直接接到第二层卷积的残差结构的求导    
    '''
    tmp_bias = np.zeros(input_img.shape[-1])
    rot_filter = rot180(filter).swapaxes(-2,-1)
    if filter.shape[0]!= 1:
        dx = conv(padding(bp_gradient,filter.shape[0]-1), rot_filter, tmp_bias)[1:-1,1:-1,:]
    else:
        dx = conv(bp_gradient, rot_filter, tmp_bias)

    return dx      

def get_derivative_conv1(input_img, filter, filter_bias, bp_gradient, conv_output):
    '''
    针对第一层卷积层的反向传播
    '''
    filter_size = filter.shape[0]
    input_img_size= input_img.shape[0]
    input_img_channal = 1 # input image 只有1个channel
    output_channal = conv_output.shape[-1]

    dw = np.zeros(filter.shape)            
    bp_gradient_filter_size = bp_gradient.shape[0]
    feature_size = input_img_size - bp_gradient_filter_size + 1
    input_img2col = np.lib.stride_tricks.as_strided(input_img, shape=(feature_size, feature_size, 1, bp_gradient_filter_size, bp_gradient_filter_size), strides=(8*input_img_channal*input_img_size, 8*input_img_channal, 8, 8*input_img_channal*input_img_size, 8*input_img_channal))
    input_img2col = np.reshape(input_img2col, (feature_size, feature_size, 1,bp_gradient_filter_size*bp_gradient_filter_size))
    bp_gradient_filter = bp_gradient.reshape(( bp_gradient_filter_size* bp_gradient_filter_size, bp_gradient.shape[-1]))
    dw = np.matmul(input_img2col,bp_gradient_filter)

    assert dw.shape == filter.shape
    db = np.sum(np.sum(bp_gradient,axis = 1),axis=0)[:None]
    assert db.shape == filter_bias.shape
    return dw,db  

def get_derivative_pool(location, bp_gradient,pool_output):
    '''
    location: 前向传播时记录的pooling最大值的位置，为一个多维array
    '''
    bp_gradient = bp_gradient.reshape(pool_output.shape)
    bp_gradient = bp_gradient.repeat(2,axis=0).repeat(2,axis=1)
    output = bp_gradient * location
    return output

def get_derivative_mfm(location, bp_gradient):
    '''
    location: 前向传播时记录的mfm最大值的位置，为一个多维array
    '''
    tmp = np.concatenate((bp_gradient,bp_gradient),axis=-1)
    output = tmp * location.astype(int) 
    return output