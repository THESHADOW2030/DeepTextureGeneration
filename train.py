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





def trainFN(disc, gen, loader, optDisc, optGen, l1, mse, epoch, writer, gScalar, dScalar, dataset, styleExtractor = None):
    loop = tqdm(loader, leave=True)
    step = 0

    styleLoss = 0

    for idx, (trainImage, fullImage, randomImage) in enumerate(loop):
        trainImage = trainImage.to("cuda" if torch.cuda.is_available() else "cpu")
        fullImage = fullImage.to("cuda" if torch.cuda.is_available() else "cpu")
        randomImage = randomImage.to("cuda" if torch.cuda.is_available() else "cpu")

        #toss a coin to decide whether to use the real image or the fake image
        coin = torch.rand(1)
        if coin < 0.5:
            image = randomImage
        else:
            image = fullImage



        with torch.cuda.amp.autocast_mode.autocast():
            fake = gen(trainImage)
            DReal = disc(image)
            DFake = disc(fake.detach())
            DRealLoss = mse(DReal, torch.ones_like(DReal))
            DFakeLoss = mse(DFake, torch.zeros_like(DFake))
            DLoss = DRealLoss + DFakeLoss

        optDisc.zero_grad()
        dScalar.scale(DLoss).backward()
        dScalar.step(optDisc)
        dScalar.update()




        with torch.cuda.amp.autocast_mode.autocast():
            DFake = disc(fake)
            GLoss = mse(DFake, torch.ones_like(DFake))  
            L1Loss = l1(fake, fullImage)
            if styleExtractor is not None:
                styleLoss = l1(styleExtractor(fake), styleExtractor(fullImage))
            GFinalLoss = GLoss + 100 * L1Loss + 50 * styleLoss                    
 

        optGen.zero_grad()
        gScalar.scale(GFinalLoss).backward()
        gScalar.step(optGen)
        gScalar.update()

        

        
        writer.add_scalar("Generator Loss", GFinalLoss, global_step=step)
        writer.add_scalar("Discriminator Loss", DLoss, global_step=step)
        step += 1

        loop.set_postfix(Disc_Loss=DLoss.item(), Gen_Loss=GFinalLoss.item(), Epoch=epoch)

   
    if epoch % 50 == 0:
        save_image(fake * 0.5 + 0.5, f"results/{dataset.trainingTarget}_{epoch}.png")
        save_image(fullImage * 0.5 + 0.5, f"results/{dataset.trainingTarget}_{epoch}_real.png")



def saveCheckpoint(model, optimizer, filename, epoch = 0):
    print("=> Saving checkpoint")
    print(f"=> Saving to {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)

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




def main(loadModel = True, train = True, saveModel = True, epochs = 100, style = False, dataName = "general"):



    res = None
    if style:
        #import resnet
        print("=> Loading resnet")
        res = resnet50(weights=ResNet50_Weights.DEFAULT).to("cuda" if torch.cuda.is_available() else "cpu")
        #remove the last layer and freeze the rest
        res = nn.Sequential(*list(res.children())[:-1])
        for param in res.parameters():
            param.requires_grad = False
        res.eval()
        print("=> Resnet loaded")


    if dataName == "general":
        checkpoints = config.weightsName.GENERAL
        if style:
            checkpoints = config.weightsName.GENERAL_STYLE
            print("=> Loading General Style")
        else:
            print("=> Loading General")

    elif dataName == "bubbly":
        checkpoints = config.weightsName.bubbly_SPECIALIZED
        print("=> Loading Bubbly")
    
    elif dataName == "fibrous":
        checkpoints = config.weightsName.fibrous_SPECIALIZED
        print("=> Loading Fibrous")

    if loadModel:
        #check if the checkpoints exist
        if not os.path.exists(checkpoints["generator"]) or not os.path.exists(checkpoints["discriminator"]):
            print("=> Checkpoints not found")
            loadModel = False
        else:
            print("=> Checkpoints found")


    




   
    discriminator = Discriminator(inChannels=3).to("cuda")
    generator = Generator(imgChannels=3, numResiduals=9).to("cuda")

    optDiscriminator = optim.Adam(discriminator.parameters(), lr=config.learningRate, betas=(0.5, 0.999))
    optGenerator = optim.Adam(generator.parameters(), lr=config.learningRate, betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    epochCheckpoint = 0
    if loadModel:
        epochCheckpoint = loadCheckpoint(checkpoints["generator"], generator, optGenerator, config.learningRate)
        loadCheckpoint(checkpoints["discriminator"], discriminator, optDiscriminator, config.learningRate)



    dataset = Textures(dataPath = "data", trainingTarget=dataName)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=True, num_workers=config.numWorkers, pin_memory=True)

  

    writer = SummaryWriter("logs")
    gScalar = torch.cuda.amp.grad_scaler.GradScaler()
    dScalar = torch.cuda.amp.grad_scaler.GradScaler()



        

        

    #print(res(torch.randn((1, 3, 256, 256)).to("cuda" if torch.cuda.is_available() else "cpu")).shape)

    #exit()

    if train:
        for epoch in range(epochs):
            trainFN(discriminator, generator, loader, optDiscriminator, optGenerator, L1, MSE, epoch + epochCheckpoint, writer, gScalar, dScalar, dataset=dataset, styleExtractor=res)
            if saveModel:
                saveCheckpoint(discriminator, optDiscriminator, checkpoints["discriminator"], epoch=epoch + epochCheckpoint)
                saveCheckpoint(generator, optGenerator, checkpoints["generator"], epoch=epoch + epochCheckpoint)





def testModel():
    

    gen = Generator(imgChannels=3, numResiduals=9).to("cuda")
    optimizer = optim.Adam(gen.parameters(), lr=config.learningRate, betas=(0.5, 0.999))
    epoch = loadCheckpoint(config.weightsName.GENERAL["generator"], gen, optimizer, config.learningRate)

    transform = transforms.Compose([
        transforms.Resize(width = 256, height = 256),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()
    ])


    gen.eval()

    for imageName in os.listdir("./test"):
        image = Image.open(os.path.join("./test", imageName)).convert('RGB')
        image = np.array(image)
        image = transform(image = image)
        image = image["image"].float()
        image = image.unsqueeze(0)
        image = image.to("cuda" if torch.cuda.is_available() else "cpu")
        fake = gen(image)
        print(fake.shape)
        save_image(fake * 0.5 + 0.5, f"results/test_{imageName.split(sep='.')[0]}_{time()}_epoch={epoch}.png")






if __name__ == "__main__":
    #testModel()
    fire.Fire(main)

    #import resnet as a feature extractor

   
