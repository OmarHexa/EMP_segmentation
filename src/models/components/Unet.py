# https://github.com/milesial/Pytorch-UNet/tree/master/unet

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Encapsulates the double convolution operation on each feature extraction level.

    This module consists of two consecutive convolutional layers, each followed by
    batch normalization and a ReLU activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of channels in the middle layer. Defaults to None,
            in which case it is set equal to out_channels.

    Example:
        >>> double_conv = DoubleConv(in_channels=64, out_channels=128)
        >>> x = torch.randn((batch_size, 64, height, width))
        >>> output = double_conv(x)
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DoubleConv model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Encapsulates downscaling with maxpooling followed by double convolution.

    This module performs downscaling using maxpooling and then applies the double convolution operation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Example:
        >>> down_module = Down(in_channels=128, out_channels=256)
        >>> x = torch.randn((batch_size, 128, height, width))
        >>> output = down_module(x)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Encapsulates upscaling followed by double convolution.

    This module performs upscaling using either bilinear interpolation or transposed convolution,
    followed by the double convolution operation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): Flag indicating whether to use bilinear interpolation
            for upscaling. Defaults to True.

    Example:
        >>> up_module = Up(in_channels=256, out_channels=128, bilinear=True)
        >>> x1 = torch.randn((batch_size, 256, height, width))
        >>> x2 = torch.randn((batch_size, 128, 2*height, 2*width))
        >>> output = up_module(x1, x2)
    """

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: Optional[bool] = True
    ) -> None:
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Up module.

        Args:
            x1 (torch.Tensor): Input tensor from decoder
            x2 (torch.Tensor): Input tensor from encoder
        Returns:
            torch.Tensor: Output tensor.
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Encapsulates the final output convolution.

    This module performs a 1x1 convolution to produce the final output.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Example:
        >>> out_conv = OutConv(in_channels=128, out_channels=1)
        >>> x = torch.randn((batch_size, 128, height, width))
        >>> output = out_conv(x)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, bilinear: Optional[bool] = True) -> None:
        """UNet architecture for semantic segmentation.

        Args:
            in_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            bilinear (bool, optional): Flag indicating whether to use bilinear interpolation
                for upscaling. Defaults to True.

        Example:
            >>> unet_model = UNET(in_channels=3, n_classes=1, bilinear=True)
            >>> x = torch.randn((batch_size, 3, height, width))
            >>> output = unet_model(x)
        """
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor from the UNet model.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self) -> None:
        """Enable checkpointing for all layers in the UNet model."""
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)


# def Test():
#     x = torch.randn((20, 3, 256, 256))
#     print(x.shape)

#     model = UNET(3, 1)
#     preds = model(x)

#     print(preds.shape)


# if __name__ == "__main__":

# Test()
