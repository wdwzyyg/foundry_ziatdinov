# -*- coding: utf-8 -*-
"""
Scripts to reproduce Ziatdinov's AIcrystalography. 
Two sets of model weights: cubic and hexagonal
Resize the input image
(set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling)
"""
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from typing import List, Union

##############
#utils
##############
def map8bit(data):
  return ((data - data.min())/(data.max() - data.min())*255).astype('int8')

class Hook():
    """
    Returns the input and output of a
    layer during forward/backward pass

    see https://www.kaggle.com/sironghuang/
        understanding-pytorch-hooks/notebook
    """
    def __init__(self, module, backward=False):
        """
        Args:
            module: torch modul(single layer or sequential block)
            backward (bool): replace forward_hook with backward_hook
        """
        if backward is False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def mock_forward(model, dims=(1, 64, 64)):
    '''Passes a dummy variable throuh a network'''
    x = torch.randn(1, dims[0], dims[1], dims[2])
    out = model(x)
    return out



class dl_image:
    '''
    Image decoder with a trained neural network
    '''
    def __init__(self, image_data, model, *args, **kwargs):
        '''
        Args:
            image_data: 2D or 3D numpy array
                image stack or a single image (all greyscale)
            model: object
                trained pytorch model (skeleton+weights)
            *param1: tuple
                new image width and height (for resizing)
        
        Kwargs:
            **nb_classes: int
                number of classes in the model
            **downsampled: int or float
                downsampling factor
            **norm: bool
                image normalization to 1
            **use_gpu: bool
                optional use of gpu device for inference
            **histogram_equalization: bool
                Equilazes image histogram
        '''
        if image_data.ndim == 2:
            image_data = np.expand_dims(image_data, axis=0)
        self.image_data = image_data
        self.model = model
        try:
            self.rs = args[0]
        except IndexError:
            self.rs = image_data.shape[1:3]
        if 'nb_classes' in kwargs:
            self.nb_classes = kwargs.get('nb_classes')
        else:
            hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
            mock_forward(model)
            self.nb_classes = [hook.output.shape for hook in hookF][-1][1]
        if 'downsampled' in kwargs:
            self.downsampled = kwargs.get('downsampled')
        else:
            hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
            mock_forward(model)
            imsize = [hook.output.shape[-1] for hook in hookF]
            self.downsampled = max(imsize)/min(imsize)
        if 'norm' in kwargs:
            self.norm = kwargs.get('norm')
        else:
            self.norm = 1
        if 'use_gpu' in kwargs:
            self.use_gpu = kwargs.get('use_gpu')
        else:
            self.use_gpu = False
        if 'histogram_equalization' in kwargs:
            self.hist_equ = kwargs.get('histogram_equalization')
        else:
            self.hist_equ = False

    def img_resize(self):
        '''Image resizing (optional)'''
        if self.image_data.shape[1:3] == self.rs:
            return self.image_data.copy()
        image_data_r = np.zeros((self.image_data.shape[0],
                                 self.rs[0], self.rs[1]))
        for i, img in enumerate(self.image_data):
            img = cv2.resize(img, (self.rs[0], self.rs[1]))
            image_data_r[i, :, :] = img
        return image_data_r
        
    def img_pad(self, *args):
        '''Pads the image if its size (w, h)
        is not divisible by 2**n, where n is a number
        of max-pooling layers in a network'''
        try:
            image_data_p = args[0]
        except IndexError:
            image_data_p = self.image_data
        # Pad image rows (height)
        while image_data_p.shape[1] % self.downsampled!= 0:
            d0, _, d2 = image_data_p.shape
            image_data_p = np.concatenate(
                (image_data_p, np.zeros((d0, 1, d2))), axis=1)
        # Pad image columns (width)
        while image_data_p.shape[2] % self.downsampled != 0:
            d0, d1, _ = image_data_p.shape
            image_data_p = np.concatenate(
                (image_data_p, np.zeros((d0, d1, 1))), axis=2)
        return image_data_p

    def hist_equalize(self, *args, number_bins=5):
        '''Histogram equalization (optional)'''
        try:
            image_data_ = args[0]
        except IndexError:
            image_data_ = self.image_data
     
        def equalize(image):
            image_hist, bins = np.histogram(image.flatten(), number_bins)
            cdf = image_hist.cumsum()
            #cdf_normalized = cdf * image_hist.max()/ cdf.max()
            image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(image.shape)

        image_data_h = np.zeros(shape=image_data_.shape)
        for i, img in enumerate(image_data_):
            img = equalize(img)
            image_data_h[i, :, :] = img

        return image_data_h

    def torch_format(self, image_data_):
        '''Reshapes and normalizes (optionally) image data
        to make it compatible with pytorch format'''
        image_data_ = np.expand_dims(image_data_, axis=1)
        if self.norm != 0:
            image_data_ = (image_data_ - np.amin(image_data_))/np.ptp(image_data_)
        image_data_ = torch.from_numpy(image_data_).float()
        return image_data_
    
    def predict(self, images):
        '''Returns probability of each pixel
           in image belonging to an atom'''
        if self.use_gpu:
            self.model.cuda()
            images = images.cuda()
        self.model.eval()
        with torch.no_grad():
            prob = self.model.forward(images)
            if self.nb_classes > 1:
                prob = torch.exp(prob)
        if self.use_gpu:
            self.model.cpu()
            images = images.cpu()
            prob = prob.cpu()
        prob = prob.permute(0, 2, 3, 1) # reshape with channel=last as in tf/keras
        prob = prob.numpy()

        return prob
    
    
    def decode(self):
        '''Make prediction'''
        image_data_ = self.img_resize()
        if self.hist_equ:
            image_data_ = self.hist_equalize(image_data_)
        image_data_ = self.img_pad(image_data_)
        image_data_torch = self.torch_format(image_data_)
        if image_data_torch.shape[0] < 20 and min(image_data_torch.shape[2:4]) < 512:
            decoded_imgs = self.predict(image_data_torch)
        else:
            n, _, w, h = image_data_torch.shape
            decoded_imgs = np.zeros((n, w, h, self.nb_classes))
            for i in range(n):
                decoded_imgs[i, :, :, :] = self.predict(image_data_torch[i:i+1])
        n_images_str = " image was " if decoded_imgs.shape[0] == 1 else " images were "
        image_data_torch = image_data_torch.permute(0, 2, 3, 1)
        images_numpy = image_data_torch.numpy()
        return images_numpy, decoded_imgs


class find_atoms:
    '''
    Transforms pixel data from decoded images
    into  a structure 'file' of atoms coordinates
    '''
    def __init__(self, decoded_imgs, threshold = 0.5, verbose = 1):
        '''
        Args:
            decoded_imgs: the output of a neural network (softmax/sigmoid layer)
            threshold: value at which the neural network output is thresholded
        '''
        if decoded_imgs.shape[-1] == 1:
            decoded_imgs_b = 1 - decoded_imgs
            decoded_imgs = np.concatenate((decoded_imgs[:, :, :, None],
                                           decoded_imgs_b[:, :, :, None]),
                                           axis=3)
        self.decoded_imgs = decoded_imgs
        self.threshold = threshold
        self.verbose = verbose
                       
    def get_all_coordinates(self, dist_edge=5):
        '''Extract all atomic coordinates in image
        via CoM method & store data as a dictionary
        (key: frame number)'''
        def find_com(image_data):
            '''Find atoms via center of mass methods'''
            labels, nlabels = ndimage.label(image_data)
            coordinates = np.array(
                ndimage.center_of_mass(image_data, labels,
                                       np.arange(nlabels)+1))
            coordinates = coordinates.reshape(coordinates.shape[0], 2)
            return coordinates

        d_coord = {}
        for i, decoded_img in enumerate(self.decoded_imgs):
            coordinates = np.empty((0, 2))
            category = np.empty((0, 1))
            # we assume that class backgrpund is always the last one
            for ch in range(decoded_img.shape[2]-1):
                _, decoded_img_c = cv2.threshold(decoded_img[:, :, ch],
                                                 self.threshold, 1, cv2.THRESH_BINARY)
                coord = find_com(decoded_img_c)
                coord_ch = self.rem_edge_coord(coord, dist_edge)
                category_ch = np.zeros((coord_ch.shape[0], 1))+ch
                coordinates = np.append(coordinates, coord_ch, axis=0)
                category = np.append(category, category_ch, axis=0)
            d_coord[i] = np.concatenate((coordinates, category), axis = 1)
        #if self.verbose == 1:
        #    print("Atomic/defect coordinates extracted")
        return d_coord

    def rem_edge_coord(self, coordinates, dist_edge):
        '''Remove coordinates at the image edges; can be applied
           to coordinates without image as well (use Image = None
           when initializing "find_atoms" class)'''
        def coord_edges(coordinates, w, h, dist_edge):
            return [coordinates[0] > w - dist_edge,
                    coordinates[0] < dist_edge,
                    coordinates[1] > h - dist_edge,
                    coordinates[1] < dist_edge]
        if self.decoded_imgs is not None:
            if self.decoded_imgs.ndim == 3:
                w, h = self.decoded_imgs.shape[0:2]
            else:
                w, h = self.decoded_imgs.shape[1:3]
        else:
            w = np.amax(coordinates[:, 0] - np.amin(coordinates[:, 0]))
            h = np.amax(coordinates[:, 1] - np.amin(coordinates[:, 1]))
        coord_to_rem = [idx for idx, c in enumerate(coordinates) if any(coord_edges(c, w, h, dist_edge))]
        coord_to_rem = np.array(coord_to_rem, dtype = int)
        coordinates = np.delete(coordinates, coord_to_rem, axis=0)
        return coordinates


################
# Net structure
################

class conv2dblock(nn.Module):
    '''
    Creates a block consisting of convolutional
    layer, leaky relu and (optionally) dropout and
    batch normalization
    '''
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1,
                 use_batchnorm=False, lrelu_a=0.01,
                 dropout_=0):
        '''
        Args:
            input_channels: number of channels in the previous/input layer
            output_channels: number of the output channels for the present layer
            kernel_size: size (in pixels) of convolutional filter
            stride: value of convolutional filter stride
            padding: value of padding at the edges
            use_batchnorm (boolean): usage of batch normalization
            lrelu_a: value of alpha parameter in leaky/paramteric ReLU activation
            dropout_: value of dropout
        '''
        super(conv2dblock, self).__init__()
        block = []
        block.append(nn.Conv2d(input_channels,
                               output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding))
        if dropout_ > 0:
            block.append(nn.Dropout(dropout_))
        block.append(nn.LeakyReLU(negative_slope=lrelu_a))
        if use_batchnorm:
            block.append(nn.BatchNorm2d(output_channels))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        '''Forward path'''
        output = self.block(x)
        return output

class dilation_block(nn.Module):
    '''
    Creates a block with dilated convolutional
    layers (aka atrous convolutions)
    '''
    def __init__(self, input_channels, output_channels,
                 dilation_values, padding_values,
                 kernel_size=3, stride=1, lrelu_a=0.01,
                 use_batchnorm=False, dropout_=0):
        '''
        Args:
            input_channels: number of channels in the previous/input layer
            output_channels: number of the output channels for the present layer
            dilation_values: list of dilation rates for convolution operation
            kernel_size: size (in pixels) of convolutional filter
            stride: value of convolutional filter stride
            padding: value of padding at the edges
            use_batchnorm (boolean): usage of batch normalization
            lrelu_a: value of alpha parameter in leaky/paramteric ReLU activation
            dropout_: value of dropout
            '''
        super(dilation_block, self).__init__()
        atrous_module = []
        for idx, (dil, pad) in enumerate(zip(dilation_values, padding_values)):
            input_channels = output_channels if idx > 0 else input_channels
            atrous_module.append(nn.Conv2d(input_channels,
                                           output_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=pad,
                                           dilation=dil,
                                           bias=True))
            if dropout_ > 0:
                atrous_module.append(nn.Dropout(dropout_))
            atrous_module.append(nn.LeakyReLU(negative_slope=lrelu_a))
            if use_batchnorm:
                atrous_module.append(nn.BatchNorm2d(output_channels))
        self.atrous_module = nn.Sequential(*atrous_module)

    def forward(self, x):
        '''Forward path'''
        atrous_layers = []
        for conv_layer in self.atrous_module:
            x = conv_layer(x)
            atrous_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(atrous_layers, dim=-1), dim=-1)

class upsample_block(nn.Module):
    '''
    Defines upsampling block performed either with
    bilinear interpolation followed by 1-by-1
    convolution or with a transposed convolution
    '''
    def __init__(self, input_channels, output_channels,
                 mode='interpolate', kernel_size=1,
                 stride=1, padding=0):
        '''
        Args:
            input_channels: number of channels in the previous/input layer
            output_channels: number of the output channels for the present layer
            mode: upsampling mode (default: 'interpolate')
            kernel_size: size (in pixels) of convolutional filter
            stride: value of convolutional filter stride
            padding: value of padding at the edges
            '''
        super(upsample_block, self).__init__()
        self.mode = mode
        self.conv = nn.Conv2d(
            input_channels, output_channels,
            kernel_size = kernel_size,
            stride = stride, padding = padding)
        self.conv_t = nn.ConvTranspose2d(
            input_channels, output_channels,
            kernel_size=2, stride=2, padding = 0)

    def forward(self, x):
        '''Defines a forward path'''
        if self.mode == 'interpolate':
            x = F.interpolate(
                x, scale_factor=2,
                mode='bilinear', align_corners=False)
            return self.conv(x)
        return self.conv_t(x)

class ResBlock(nn.Module):
    """
    Builds a residual block
    """
    def __init__(self, nb_filters_in=20, nb_filters_out=40, lrelu_a=0,
                use_batchnorm=False):
        """
        Args:
            nb_filters_in (int): number of channels in the block input
            nb_filters_out (int): number of channels in the block output
            lrelu_a=0 (float): negative slope value for leaky ReLU
        """
        super(ResBlock, self).__init__()
        self.lrelu_a = lrelu_a
        self.use_batchnorm = use_batchnorm
        self.c0 = nn.Conv2d(nb_filters_in,
                            nb_filters_out,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.c1 = nn.Conv2d(nb_filters_out,
                           nb_filters_out,
                           kernel_size=3,
                           stride=1,
                           padding=1)
        self.c2 = nn.Conv2d(nb_filters_out,
                           nb_filters_out,
                           kernel_size=3,
                           stride=1,
                           padding=1)
        self.bn1 = nn.BatchNorm2d(nb_filters_out)
        self.bn2 = nn.BatchNorm2d(nb_filters_out)

    def forward(self, x):
        """Defines forward path"""
        x = self.c0(x)
        residual = x
        out = self.c1(x)
        if self.use_batchnorm:
            out = self.bn1(out)
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        out = self.c2(out)
        if self.use_batchnorm:
            out = self.bn2(out)
        out += residual
        out = F.leaky_relu(out, negative_slope=self.lrelu_a)
        return out

class ResModule(nn.Module):
    """
    Stitches multiple convolutional blocks together
    """
    def __init__(self, input_channels, output_channels, res_depth, lrelu_a=0.01, use_batchnorm=False):
        """
        Args:
            input_channels (int): number of filters in the input layer
            output_channels (int): number of channels in the output layer
            res_depth (int): number of residual blocks in a residual module
        """
        super(ResModule, self).__init__()
        res_module = []
        for i in range(res_depth):
            input_channels = output_channels if i > 0 else input_channels
            res_module.append(
                ResBlock(input_channels, output_channels, lrelu_a=lrelu_a, use_batchnorm=use_batchnorm))
        self.res_module = nn.Sequential(*res_module)

    def forward(self, x):
        """Defines a forward path"""
        x = self.res_module(x)
        return x


class atomsegnet(nn.Module):
    '''
    Builds  a fully convolutional neural network model
    '''
    def __init__(self, nb_classes=1, nb_filters=32):
        '''
        Args:
            nb_filters: number of filters in the first convolutional layer
        '''
        super(atomsegnet, self).__init__()
        self.pxac = 'sigmoid' if nb_classes < 2 else 'softmax'
        self.c1 = conv2dblock(1, nb_filters)
        
        self.c2 = nn.Sequential(conv2dblock(nb_filters,
                                            nb_filters*2),
                                conv2dblock(nb_filters*2,
                                            nb_filters*2))
        
        self.c3 = nn.Sequential(conv2dblock(nb_filters*2,
                                            nb_filters*4,
                                            dropout_=0.3),
                                conv2dblock(nb_filters*4,
                                            nb_filters*4,
                                            dropout_=0.3))
        
        self.bn = dilation_block(nb_filters*4,
                                 nb_filters*8,
                                 dilation_values=[2, 4, 6],
                                 padding_values=[2, 4, 6],
                                 dropout_=0.5)
        
        self.upsample_block1 = upsample_block(nb_filters*8,
                                              nb_filters*4)
        
        self.c4 = nn.Sequential(conv2dblock(nb_filters*8,
                                            nb_filters*4,
                                            dropout_=0.3),
                                conv2dblock(nb_filters*4,
                                            nb_filters*4,
                                            dropout_=0.3))
        
        self.upsample_block2 = upsample_block(nb_filters*4,
                                              nb_filters*2)
        
        self.c5 = nn.Sequential(conv2dblock(nb_filters*4,
                                            nb_filters*2),
                                conv2dblock(nb_filters*2,
                                            nb_filters*2))
        
        self.upsample_block3 = upsample_block(nb_filters*2,
                                              nb_filters)
        
        self.c6 = conv2dblock(nb_filters*2,
                              nb_filters)
        
        self.px = nn.Conv2d(nb_filters,
                            nb_classes,
                            kernel_size=1,
                            stride=1,
                            padding=0)
               
    def forward(self, x):
        '''Defines a forward path'''
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.c2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.c3(d2)
        d3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        # Atrous convolutions
        bn = self.bn(d3)
        # Expanding path
        u3 = self.upsample_block1(bn)
        u3 = torch.cat([c3, u3], dim=1)
        u3 = self.c4(u3)
        u2 = self.upsample_block2(u3)
        u2 = torch.cat([c2, u2], dim=1)
        u2 = self.c5(u2)
        u1 = self.upsample_block3(u2)
        u1 = torch.cat([c1, u1], dim=1)
        u1 = self.c6(u1)
        # pixel-wise classification
        px = self.px(u1)
        if self.pxac == 'sigmoid':
            output = torch.sigmoid(px)
        elif self.pxac == 'softmax':
            output = F.log_softmax(px, dim=1)
        return output
    

class resatomsegnet_s2(nn.Module):
    '''Builds  a fully convolutional neural network model'''
    def __init__(self, nb_classes=1, nb_filters=32):
        '''
        Args:
            nb_classes (int): number of classes to be predicted
            nb_filters (int): number of filters in the first convolutional layer
        '''
        super(resatomsegnet_s2, self).__init__()
        self.pxac = 'softmax' if nb_classes > 1 else 'sigmoid'
        self.c1 = conv2dblock(1, nb_filters)
        self.c2 = ResModule(nb_filters, nb_filters*2, res_depth=2)
        self.bn = ResModule(nb_filters*2, nb_filters*4, res_depth=2)
        self.upsample_block1 = upsample_block(nb_filters*4, nb_filters*2)
        self.c3 = ResModule(nb_filters*4, nb_filters*2, res_depth=2) 
        self.upsample_block2 = upsample_block(nb_filters*2, nb_filters)
        self.c4 = conv2dblock(nb_filters*2, nb_filters)
        self.px = nn.Conv2d(nb_filters, nb_classes, kernel_size = 1, stride = 1, padding = 0)
    
    def forward(self, x):
        '''Defines a forward path'''
        # Contracting path
        c1 = self.c1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)      
        c2 = self.c2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)      
        # Bottleneck 
        bn = self.bn(d2)
        # Expanding path
        u2 = self.upsample_block1(bn)
        u2 = torch.cat([c2, u2], dim = 1)
        u2 = self.c3(u2)
        u1 = self.upsample_block2(u2)
        u1 = torch.cat([c1, u1], dim = 1)
        u1 = self.c4(u1)
        # pixel-wise classification
        px = self.px(u1)
        if self.pxac == 'sigmoid':
            output = torch.sigmoid(px)
        elif self.pxac == 'softmax':
            output = F.log_softmax(px, dim=1)
        return output

##################################
#load model and main function: run
##################################
def load_torchmodel(lattice_type):
    '''
    Loads saved weights into a model for a specific lattice type

    Args:
        lattice_type: str
            Select between ("hexagonal" and "cubic")
    
    Returns:
        pytorch model with pretrained weights loaded
    '''
    #device_ = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    ##############################
    device_ = torch.device('cpu')###
    ##############################
    if lattice_type == 'hexagonal':
        model = atomsegnet()
        checkpoint = torch.load(
            'G-Si-DFT0-1-4-best_weights.pt', map_location=device_
        )
    elif lattice_type == 'cubic':
        model = resatomsegnet_s2()
        checkpoint = torch.load(
            'cubic-best_weights.pt', map_location=device_)
    else:
        raise ValueError(
            'Select one of the currently available models: hexagonal, cubic'
        )
    model.load_state_dict(checkpoint)
    return model

def run(inputdict,*args):
  imarray_original = inputdict['image']
  modelname = inputdict['modelweights']
  change_size = inputdict['change_size']
  # Input resize
  ori_image = Image.fromarray(map8bit(imarray_original), 'L')
  width, height = ori_image.size   

  if change_size == 1:
    ori_content = ori_image
  elif (change_size > 1):
    # Upsample using bicubic
    ori_content = ori_image.resize((int(width *change_size), int(height * change_size)), Image.BICUBIC)
  else:
    # downsample using bilinear
    ori_content = ori_image.resize((int(width * change_size), int(height * change_size)), Image.BILINEAR)
  # load model
  model = load_torchmodel(modelname)
  # Apply a trained model to the loaded data
  img, dec = dl_image(np.asarray(ori_content), model).decode()
  # Get atomic coordinates:
  coord = find_atoms(dec).get_all_coordinates()    
  res = coord[0][:,:2].T
  res[[0,1]] = res[[1,0]]
  return res

def inference(dicts: Union[List[dict], dict]):
    """
    Run a model to predict atom column coordinates in the input STEM images
    modelweight: cubic or hexagonal
    change_size: set as 1 to remain the same, set greater than 1 to upsampling and smaller than 1 to downsampling

    Example:

    >>> # Make a dict of your inputs
    >>> # use a list of dicts if you want to run it though multiple images at one time
    >>> dict = {'image': array, 'modelweights':'hexagonal','change_size': 1} 

    """
    results = []
    if isinstance(dicts, list):
      for i in range(len(dicts)):
        results.append(run(dicts[i]))
      return results  
    elif isinstance(dicts, dict):
        results.append(run(dicts))
        return results
    else: raise AssertionError("Input should be dict or list of dict")