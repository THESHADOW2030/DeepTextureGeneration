import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Textures


from discriminator import Discriminator
from generator import Generator

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from time import time

import numpy as np


from PIL import Image
import fire
import albumentations as transforms
from albumentations.pytorch import ToTensorV2
import os




import config

from torchvision.models import resnet50, ResNet50_Weights

import torchvision



def loadCheckpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    print(f"=> Loading from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return checkpoint["epoch"]



def testModel():
    
    checkpoints = config.weightsName.grassWithRocks_HighlySpecialized    

    gen = Generator(imgChannels=3, numResiduals=9).to("cuda")
    optimizer = optim.Adam(gen.parameters(), lr=config.learningRate, betas=(0.5, 0.999))
    epoch = loadCheckpoint(checkpoints["generator"], gen, optimizer, config.learningRate)
    gen.eval()

    #path = "/home/shadow2030/Documents/deepLearning/DeepLearningProject/test/stripped_3.jpg"
    path = "/home/shadow2030/Documents/deepLearning/DeepLearningProject/data/striped_0035.jpg"
    #get random image
    image = Image.open(path)
    

    image = image.convert('RGB')
    image = np.array(image)

    fullImage = Image.open(path).convert('RGB')
    tmpTrans = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]),
        transforms.Resize(height = 512, width = 512),
        ToTensorV2()
    ])
    fullImage = tmpTrans(image = np.array(fullImage))["image"].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    #crop the real image at a random location
    trans = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]),
        #random crop
        transforms.RandomCrop(width=256, height=256),
        ToTensorV2()
    ])

    image = trans(image = np.array(image))["image"].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    tmp = image.float()  


    output = gen(tmp) * 0.5 + 0.5
    

    
    #pad the image to be 512
    image = torchvision.transforms.Pad(padding= (128, 128, 128, 128),padding_mode="constant")(image)

    save_image(torch.cat((image, fullImage, output)  , dim=3), f"./tmp/{checkpoints['dataName']}_test_{epoch}.png")






def testRandomNoise(ganType = "", path = "./weights"):

    checkpoints = config.weightsName.striped_SPECIALIZED

    ganType = checkpoints["dataName"]
    #imagePath = "/home/shadow2030/Documents/deepLearning/DeepLearningProject/data/striped_0035.jpg"
    imagePath = "/home/shadow2030/Documents/deepLearning/DeepLearningProject/test/stripped_4.jpg"
    image = Image.open(imagePath).convert('RGB')
    image = np.array(image)


    imagePath2 = "/home/shadow2030/Documents/deepLearning/DeepLearningProject/data/striped_0081.jpg"
    image2 = Image.open(imagePath2).convert('RGB')
    image2 = np.array(image2)

    #crop the image 256x256
    trans = transforms.Compose([
        transforms.Resize(width=512, height=512),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]),
        #random crop
        transforms.RandomCrop(width=256, height=256),
        ToTensorV2()
    ])

    image = trans(image = np.array(image))["image"].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    image2 = trans(image = np.array(image2))["image"].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    coef = 1

    #random noise
    #image = torch.normal(0, 1, size=image.shape).to("cuda" if torch.cuda.is_available() else "cpu")
    #image2 = torch.normal(0, 1, size=image.shape).to("cuda" if torch.cuda.is_available() else "cpu")

    #add random noise
    image = image * coef + image2 * (1 - coef)


    
    gen = Generator(imgChannels=3, numResiduals=9).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(gen.parameters(), lr=config.learningRate, betas=(0.5, 0.999))
    epoch = loadCheckpoint(checkpoints["generator"], gen, optimizer, config.learningRate)
    gen.eval()

    output = gen(image) * 0.5 + 0.5


    save_image(output, f"./tmp/randomNoise_{ganType}_test_{epoch + 1}.png")
    print(f"=> Saved to ./tmp/randomNoise_{ganType}_test_{epoch + 1}.png")

  


def testHighResModel():

    checkpoints = config.weightsName.striped_SPECIALIZED

    #load the image
    imagePath = checkpoints["highResImagePath"]
    image = Image.open(imagePath).convert('RGB')
    image = np.array(image)

    #randomlySample the image
    trans = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]),
        #random crop
        transforms.RandomCrop(width=256, height=256),
        ToTensorV2()
    ])

    images = [trans(image = np.array(image))["image"].unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu") for i in range(10)]

    gen = Generator(imgChannels=3, numResiduals=9).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(gen.parameters(), lr=config.learningRate, betas=(0.5, 0.999))
    epoch = loadCheckpoint(checkpoints["generator"], gen, optimizer, config.learningRate)
    gen.eval()

    for i, image in enumerate(images):
        output = gen(image) 
        save_image(torch.cat((torchvision.transforms.Pad(padding= (128, 128, 128, 128),padding_mode="constant", fill=255)(image) * 0.5 + 0.5, output* 0.5 + 0.5)  , dim=3), f"./finalImages/conditioned_timber_{i}.png")
        print(f"=> Saved to ./finalImages/conditioned_timber_{i}.png")

        #random noise
        image = torch.randn((1, 3, 256, 256)).to("cuda" if torch.cuda.is_available() else "cpu")
        #normalize the image with std and mean of 0.5
        image = (image - 0.5) / 0.5
        output = gen(image) * 0.5 + 0.5
        save_image(output, f"./finalImages/random_grassWithRocks_{i}.png")
        print(f"=> Saved to ./finalImages/random_grassWithRocks_{i}.png")





if __name__ == "__main__":
    pass
