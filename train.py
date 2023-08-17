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




def trainFN(disc, gen, loader, optDisc, optGen, l1, mse, epoch, writer, gScalar, dScalar, dataset):
    loop = tqdm(loader, leave=True)
    step = 0


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
            GFinalLoss = GLoss + 100 * L1Loss

        optGen.zero_grad()
        gScalar.scale(GFinalLoss).backward()
        gScalar.step(optGen)
        gScalar.update()

        

        
        writer.add_scalar("Generator Loss", GFinalLoss, global_step=step)
        writer.add_scalar("Discriminator Loss", DLoss, global_step=step)
        step += 1

        loop.set_postfix(Disc_Loss=DLoss.item(), Gen_Loss=GFinalLoss.item(), Epoch=epoch)

    #with probability 1/1000 save the images
    if epoch % 10 == 0:
        save_image(fake * 0.5 + 0.5, f"results/{epoch}.png")
        save_image(fullImage * 0.5 + 0.5, f"results/{epoch}_real.png")



def saveCheckpoint(model, optimizer, filename, epoch = 0):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)

def loadCheckpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return checkpoint["epoch"]




def main(loadModel = True, train = True, saveModel = True, epochs = 100):


   
    discriminator = Discriminator(inChannels=3).to("cuda")
    generator = Generator(imgChannels=3, numResiduals=9).to("cuda")

    optDiscriminator = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optGenerator = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()


    




    epochCheckpoint = 0
    if loadModel:
        epochCheckpoint = loadCheckpoint("discriminatorWeights.pth.tar", discriminator, optDiscriminator, 2e-4)
        loadCheckpoint("generatorWeights.pth.tar", generator, optGenerator, 2e-4)


    dataset = Textures(dataPath = "data")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

  

    writer = SummaryWriter("logs")

    gScalar = torch.cuda.amp.grad_scaler.GradScaler()
    dScalar = torch.cuda.amp.grad_scaler.GradScaler()

    if train:
        for epoch in range(epochs):
            trainFN(discriminator, generator, loader, optDiscriminator, optGenerator, L1, MSE, epoch + epochCheckpoint, writer, gScalar, dScalar, dataset=dataset)
            if saveModel:
                saveCheckpoint(discriminator, optDiscriminator, "discriminatorWeights.pth.tar", epoch=epoch + epochCheckpoint)
                saveCheckpoint(generator, optGenerator, "generatorWeights.pth.tar", epoch=epoch + epochCheckpoint)





def testModel():
    

    gen = Generator(imgChannels=3, numResiduals=9).to("cuda")
    optimizer = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loadCheckpoint("generatorWeights.pth.tar", gen, optimizer, 2e-4)

    gen.eval()

    image = Image.open("./test/1.jpeg").convert('RGB')
    image = np.array(image)


    transform = transforms.Compose([
                        transforms.CenterCrop(width = 256, height = 256),
                        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value=255),
                        ToTensorV2()
                    ])
    
    image = transform(image = image)
    image = image["image"].float()
    image = image.unsqueeze(0).to("cuda")

    with torch.no_grad():
        fake = gen(image)
        save_image(fake * 0.5 + 0.5, f"results/test1.png")






if __name__ == "__main__":
    #testModel()
    fire.Fire(main)
