#https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet


import torch
import torchvision
import torch.nn as nn

# encapsulate the double convulation operation on each feature extraction level

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channels,mid_channels=None) -> None:
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels=out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,mid_channels,3,1,1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(), 
            nn.Conv2d(mid_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class UNETBilinear(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNETBilinear,self).__init__()
        self.features = [64,128,256,512]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # creates the operation chain from the input to the lowest part(encoder) of the model 
        for feature in self.features:
            self.down.append(DoubleConv(in_channels,feature))
            in_channels =feature

        """ Upsampling can't increase the feature number we need to increase the feature by a 
        additional step of convulation. so instead of convTransposed we do upsampling and conv2d.
        In the double conv opeariton we added midchannel to solve this requirement of one extra
        conv2d to increase the channel number."""
        for feature in reversed(self.features):
            self.up.append(nn.UpsamplingBilinear2d(scale_factor=2))
            # DoubleConv without mid-channel at the last stage.
            if feature!=self.features[0]:
                self.up.append(DoubleConv(feature*2,feature//2,feature))
            else:
                self.up.append(DoubleConv(feature*2,feature))

        # no increase in the number of channels in the bottom feature extraction part
        self.bottom = DoubleConv(self.features[-1],self.features[-1])
        self.final_conv = nn.Conv2d(self.features[0],out_channels,kernel_size=1)
        # self.fixer= DoubleConv(self.features[0]*2,self.features[0])

    def forward(self,x):
        # To store the feature vector on each layer of down sampling
        skip_connection =[]

        for down in self.down:
            x = down(x)
            skip_connection.append(x)
            x =self.pool(x)
        
        x= self.bottom(x)# no. of feature is 512 (unlike the original paper)

        skip_connection = list(reversed(skip_connection))

        for idx in range(0,len(self.up),2):
            x = self.up[idx](x)
            connection = skip_connection[idx//2] # beacuse idx increment is 2.
            #simplest way to solve for size mismatch of the two vector. 
            #TODO look for other method.
            if x.shape!= connection.shape:
                x= torchvision.transforms.functional.resize(x,size = connection.shape[2:])
            x= torch.concat((x,connection),dim=1)
            x = self.up[idx+1](x)

        return self.final_conv(x)




class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNET,self).__init__()
        self.features = [64,128,256,512]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        for feature in self.features:
            self.down.append(DoubleConv(in_channels,feature))
            in_channels =feature

        for feature in reversed(self.features):
            self.up.append(nn.ConvTranspose2d(2*feature,feature,kernel_size=2,stride=2))
            self.up.append(DoubleConv(feature*2,feature))

        self.bottom = DoubleConv(self.features[-1],self.features[-1]*2)
        self.final_conv = nn.Conv2d(self.features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connection =[]

        for down in self.down:
            x = down(x)
            skip_connection.append(x)
            x =self.pool(x)
        
        x= self.bottom(x)

        skip_connection = list(reversed(skip_connection))

        for idx in range(0,len(self.up),2):
            x = self.up[idx](x)
            connection = skip_connection[idx//2]
            if x.shape!= connection.shape:
                x= torchvision.transforms.functional.resize(x,size = connection.shape[2:])
            x= torch.concat((x,connection),dim=1)
            x = self.up[idx+1](x)

        return self.final_conv(x)



def Test():
    x = torch.randn((3,3,160,240))
    model =UNET(3,1)
    preds =model(x)

    print(preds.shape)
    print(x.shape)





if __name__ == "__main__":
    Test()