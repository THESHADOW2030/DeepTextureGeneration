import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, down=True, use_act = True, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, **kwargs) if down else nn.ConvTranspose2d(inChannels, outChannels, **kwargs),
            nn.InstanceNorm2d(outChannels), 
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channcels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(ConvBlock(channcels, channcels, kernel_size = 3, padding = 1),
                                   ConvBlock(channcels, channcels, use_act = False, kernel_size = 3, padding = 1)
                                    )
        
    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self, imgChannels, numFeatures = 64, numResiduals = 9, finalSize = 512):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(imgChannels, numFeatures, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.downBlocks = nn.ModuleList([   #for downsampling
            ConvBlock(numFeatures, numFeatures*2, kernel_size = 3, stride = 2, padding = 1),
            ConvBlock(numFeatures*2, numFeatures*4,kernel_size = 3, stride = 2, padding = 1 )
            
        ])

        self.residualBlocks = nn.Sequential(*[ResidualBlock(numFeatures*4) for _ in range(numResiduals)])

        self.convBlock512 = ConvBlock(numFeatures*4, numFeatures*8, kernel_size = 3, stride = 1, padding = 1) 

        

        #upsampling to double the size (finalSize)
        self.upBlocks = nn.ModuleList([
            ConvBlock(numFeatures*8, numFeatures*4, down=False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            ConvBlock(numFeatures*4, numFeatures * 2, down=False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            ConvBlock(numFeatures*2, numFeatures, down=False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        ])

        self.last = nn.Conv2d(numFeatures, imgChannels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        
        

    def forward(self, x):
        #print("Input: ", x.shape)              #torch.Size([1, 3, 256, 256])
        x = self.initial(x)
        #print("After initial: ", x.shape)      #torch.Size([1, 64, 256, 256])
        
        for layer in self.downBlocks:
            x = layer(x)
        
        skipConnection = x
        #print("After down: ", x.shape)     #torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64])
        x = self.residualBlocks(x)
        #print("After residual: ", x.shape)     #torch.Size([1, 256, 64, 64])
        x = torch.cat([x, skipConnection], dim=1)
        #print("After skipConnection: ", x.shape) #torch.Size([1, 512, 64, 64])
        for layer in self.upBlocks:
            x = layer(x) 
            #print("After up: ", x.shape)       #torch.Size([1, 64, 512, 512])
        x = self.last(x)
        #print("After last: ", x.shape)         #torch.Size([1, 3, 512, 512])
        return torch.tanh(x)
    

#PRINTS OF THE SIZE OF THE TENSORS
"""
Input:  torch.Size([1, 3, 256, 256])
After initial:  torch.Size([1, 64, 256, 256])
After down:  torch.Size([1, 128, 128, 128])
After down:  torch.Size([1, 256, 64, 64])
After residual:  torch.Size([1, 256, 64, 64])
After convBlock512:  torch.Size([1, 512, 64, 64])
After up:  torch.Size([1, 256, 128, 128])
After up:  torch.Size([1, 128, 256, 256])
After up:  torch.Size([1, 64, 512, 512])
After last:  torch.Size([1, 3, 512, 512])
 """
    
    



def test():
    imgChannels = 3
    imgSize = 256
    x = torch.randn((2, imgChannels, imgSize, imgSize))
    gen  = Generator(imgChannels, numFeatures=64, numResiduals=9, finalSize=256)
   
    print(gen(x).shape)


if __name__ == "__main__":
    test()