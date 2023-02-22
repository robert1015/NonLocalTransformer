
import torch
import torch.nn as nn

class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()
        self.outdim = channels

    def forward(self, x ):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x) # x is what i want
        channel_weight = x
        x = module_input * x
        """
        # Spatial attention module
        module_input = input
        avg = torch.mean(input, 1, True)
        mx, _ = torch.max(input, 1, True)
        input = torch.cat((avg, mx), 1)
        input = self.conv_after_concat(input)
        input = self.sigmoid_spatial(input)
        output = module_input * input
        """
        #del avg
        #del mx

        return x ,channel_weight