import torch
from torch import nn
from torchvision.models import regnet_y_400mf, RegNet_Y_400MF_Weights
from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
from torchvision.models import regnet_x_400mf, RegNet_X_400MF_Weights
from torchvision.models import regnet_x_800mf, RegNet_X_800MF_Weights

"""Adapted from https://github.com/mareksubocz/TrackNet"""


class TrackNetV2Base(nn.Module):
    def _make_convolution_sublayer(self, in_channels, out_channels, dropout_rate=0.0):
        layer = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        ]
        if dropout_rate > 1e-15:
            #print('!'*50, 'dropout used!')
            layer.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layer)

    def _make_convolution_layer(self, in_channels, out_channels, num, dropout_rate=0.0):
        layers = []
        layers.append(self._make_convolution_sublayer(in_channels, out_channels, dropout_rate=dropout_rate))
        for _ in range(num-1):
            layers.append(self._make_convolution_sublayer(out_channels, out_channels, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def __init__(self, grayscale=False, dropout=0.1, sequence_length=3):
        """Backbone of TrackNetV2, until the last 64 channel convolution of DeconvNet.

        Parameters
        ----------
        grayscale : bool, optional
            if True, the input frames are converted to grayscale. By default False
        dropout : float, optional
            choose whether to use dropout, by default 0.1
        sequence_length : int, optional
            sequence of input frames, by default 3
        """
        super().__init__()

        self.sequence_length = sequence_length

        # VGG16
        if grayscale:
            self.vgg_conv1 = self._make_convolution_layer(sequence_length, 64, 2, dropout_rate=dropout)
        else:
            self.vgg_conv1 = self._make_convolution_layer(3*sequence_length, 64, 2, dropout_rate=dropout)
        self.vgg_maxpool1 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv2 = self._make_convolution_layer(64, 128, 2, dropout_rate=dropout)
        self.vgg_maxpool2 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv3 = self._make_convolution_layer(128, 256, 3, dropout_rate=dropout)
        self.vgg_maxpool3 = nn.MaxPool2d((2,2), stride=(2,2))
        self.vgg_conv4 = self._make_convolution_layer(256, 512, 3, dropout_rate=dropout)

        # Deconv / UNet
        self.unet_upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv1 = self._make_convolution_layer(768, 256, 3, dropout_rate=dropout)
        self.unet_upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv2 = self._make_convolution_layer(384, 128, 2, dropout_rate=dropout)
        self.unet_upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.unet_conv3 = self._make_convolution_layer(192, 64, 2, dropout_rate=dropout)

    def forward(self, x):
        # VGG16
        x1 = self.vgg_conv1(x)
        x = self.vgg_maxpool1(x1)
        x2 = self.vgg_conv2(x)
        x = self.vgg_maxpool2(x2)
        x3 = self.vgg_conv3(x)
        x = self.vgg_maxpool3(x3)
        x = self.vgg_conv4(x)

        # Deconv / UNet
        x = torch.concat([self.unet_upsample1(x), x3], dim=1)
        x = self.unet_conv1(x)
        x = torch.concat([self.unet_upsample2(x), x2], dim=1)
        x = self.unet_conv2(x)
        x = torch.concat([self.unet_upsample3(x), x1], dim=1)
        x = self.unet_conv3(x)

        return x

    def save(self, path, whole_model=False):
        if whole_model:
            torch.save(self, path)
        else:
            torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device)['model_state_dict'])


class TrackNetV2MSE(TrackNetV2Base):
    def __init__(self, one_output_frame=True, *args, **kwargs):
        """TrackNetV2 implementation, where the last layer outputs a single heatmap.
        Meant to be used with MSE loss.

        Parameters
        ----------
        one_output_frame : bool, optional
            if set to True, outputs a single heatmap. If set to False, outputs a heatmap for each input frame.
            By default True
        *args, **kwargs : passed to `TrackNetV2Base`
        """
        super().__init__(*args, **kwargs)

        if one_output_frame:
            self.last_conv = nn.Conv2d(64, 1, kernel_size=(1,1), padding="same")
        else:
            self.last_conv = nn.Conv2d(64, self.sequence_length, kernel_size=(1,1), padding="same")
        self.last_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)

        x = self.last_conv(x)
        x = self.last_sigmoid(x)

        return x


class TrackNetV2NLL(TrackNetV2Base):
    def __init__(self, *args, **kwargs):
        """`TrackNetV2` implementation, where the last layer outputs a single heatmap.
        Meant to be used with NLL loss along the channel dimension.

        Parameters
        ----------
        *args, **kwargs : passed to TrackNetV2Base
        """
        super().__init__(*args, **kwargs)

        self.last_conv = nn.Conv2d(64, 256, kernel_size=(1,1), padding="same")
        self.last_logsoftmax = nn.LogSoftmax(dim=1) # compute logsoftmax along the channel dimension

    def forward(self, x):
        x = super().forward(x)

        x = self.last_conv(x)
        x = self.last_logsoftmax(x)

        return x


"""RegNetX and RegNetY taken from the following paper:
    Designing Network Design Spaces (https://arxiv.org/abs/2003.13678)"""


def my_regnet_y_400mf(sequence_length=3, grayscale=False, pretrained=True):
    if pretrained:
        model = regnet_y_400mf(weights=RegNet_Y_400MF_Weights)
        model.fc = nn.Linear(in_features=440, out_features=2)
    else:
        model = regnet_y_400mf(num_classes=2)

    channel_mult = 1 if grayscale else 3
    model.stem[0] = nn.Conv2d(sequence_length*channel_mult, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return model


def my_regnet_y_800mf(sequence_length=3, grayscale=False, pretrained=True):
    if pretrained:
        model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights)
        model.fc = nn.Linear(in_features=784, out_features=2)
    else:
        model = regnet_y_800mf(num_classes=2)

    channel_mult = 1 if grayscale else 3
    model.stem[0] = nn.Conv2d(sequence_length*channel_mult, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return model


def my_regnet_x_400mf(sequence_length=3, grayscale=False, pretrained=True):
    if pretrained:
        model = regnet_x_400mf(weights=RegNet_X_400MF_Weights)
        model.fc = nn.Linear(in_features=400, out_features=2)
    else:
        model = regnet_x_400mf(num_classes=2)

    channel_mult = 1 if grayscale else 3
    model.stem[0] = nn.Conv2d(sequence_length*channel_mult, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return model


def my_regnet_x_800mf(sequence_length=3, grayscale=False, pretrained=True):
    if pretrained:
        model = regnet_y_800mf(weights=RegNet_X_800MF_Weights)
        model.fc = nn.Linear(in_features=672, out_features=2)
    else:
        model = regnet_y_800mf(num_classes=2)

    channel_mult = 1 if grayscale else 3
    model.stem[0] = nn.Conv2d(sequence_length*channel_mult, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return model
