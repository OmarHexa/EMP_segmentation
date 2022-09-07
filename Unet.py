#https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet


import torch
import torchvision
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channels) -> None:
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channels,3,1,1,bias=False),
            nn.ReLU(), 
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        return self.conv(x)

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