## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# We created a custom function for linear activation function defined
# as follows:
# f(x) = x 

# simply define a silu function
def linear_activation(input):
    '''
    Applies the linear activation function:
        linear_activation(x) = x 
    '''
    return input 

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class LinearActivation(nn.Module):
    '''
    Applies the linear activation function element-wise:
        LinearActivation(x) = x 
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return linear_activation(input) 



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel
        
        # We adapted the architecture proposed in the following paper 
        # https://arxiv.org/pdf/1710.00977.pdf                
        
        # 1 input image channel (grayscale), 32 output channels/feature maps 
        # 6x6 square convolution kernel 
        ## output size = (W-F)/S +1 = (224-6)/1 +1 = 219 
        # the output Tensor for one image, will have the dimensions: (8, 219, 219)
        # after one pool layer, this becomes (8, 109, 109)
        
        self.conv1 = nn.Conv2d(1, 8, 6) 
        
        
        self.conv1_bn = nn.BatchNorm2d(8)
                
        # maxpool layer
        # pool with kernel_size=2, stride=2

        self.pool = nn.MaxPool2d(2, 2)
        
        
        # second conv layer: 8 inputs, 16 outputs, 5x5 conv 
        ## output size = (W-F)/S +1 = (109-5)/1 +1 = 105 
        # the output Tensor for one image, will have the dimensions: (16, 105, 105)
        # after another pool layer this becomes (16, 52,  52)
        self.conv2 = nn.Conv2d(8, 16, 5) 
        
        self.conv2_bn = nn.BatchNorm2d(16)
                        
        
        # third conv layer: 16 inputs, 32 outputs, 4x4 conv 
        ## output size = (W-F)/S +1 = (52-4)/1 +1 = 49
        # the output Tensor for one image, will have the dimensions: (32, 49, 49)
        # after another pool layer this becomes (32, 24, 24)
        self.conv3 = nn.Conv2d(16, 32, 4) 
        
        
        self.conv3_bn = nn.BatchNorm2d(32)
        
                
        # fourth conv layer: 32 inputs, 64 outputs, 3x3 conv 
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        # the output Tensor for one image, will have the dimensions: (64, 22, 22)
        # after another pool layer this becomes (64, 11, 11)
        self.conv4 = nn.Conv2d(32, 64, 3) 
        
        self.conv4_bn = nn.BatchNorm2d(64)
        

        # fifth conv layer: 64 inputs, 128 outputs, 2x2 conv 
        ## output size = (W-F)/S +1 = (11-2)/1 +1 = 10
        # the output Tensor for one image, will have the dimensions: (128, 10, 10)
        # after another pool layer this becomes (128, 5, 5)
        self.conv5 = nn.Conv2d(64, 128, 2) 
        
        self.conv5_bn = nn.BatchNorm2d(128)
        
        
        # sixth conv layer: 128 inputs, 256 outputs, 1x1 conv 
        ## output size = (W-F)/S +1 = (5-1)/1 +1 = 5
        # the output Tensor for one image, will have the dimensions: (256, 5, 5)
        # after another pool layer this becomes (256, 2, 2)
        self.conv6 = nn.Conv2d(128, 256, 1) 

        
        self.conv6_bn = nn.BatchNorm2d(256)
        
                                               
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.drop1 = nn.Dropout(p=0.05)                       
        self.drop2 = nn.Dropout(p=0.10)                       
        self.drop3 = nn.Dropout(p=0.15) 
        self.drop4 = nn.Dropout(p=0.20) 
        self.drop5 = nn.Dropout(p=0.25) 
        self.drop6 = nn.Dropout(p=0.30) 
        self.drop7 = nn.Dropout(p=0.35) 
        self.drop8 = nn.Dropout(p=0.40) 
        
        
         
        self.fc1 = nn.Linear(1024, 1000)
        self.fc2 = nn.Linear(1000, 1000) 
        self.fc3 = nn.Linear(1000, 136) 
                               
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        # initialize activation function
        activation_function = LinearActivation()
        
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)        
        
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)                
        x = self.pool(x)
        x = self.drop2(x)                
        
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop3(x)
        
        
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop4(x)        
        

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop5(x)
        

        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop6(x)
                
        # Flatten Layer 
        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.drop7(x)
        
        x = self.fc2(x)
        x = activation_function(x)
        x = self.drop8(x)
        x = self.fc3(x)
        
        return x