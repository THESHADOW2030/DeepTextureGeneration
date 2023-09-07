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






def trainFN(disc, gen, loader, optDisc, optGen, l1, mse, epoch, writer, gScalar, dScalar, dataset, styleExtractor = None):
    loop = tqdm(loader, leave=True)
    step = 0

    contentLoss = 0

    for idx, (trainImage, fullImage, randomImage) in enumerate(loop):
        trainImage = trainImage.to("cuda" if torch.cuda.is_available() else "cpu")
        fullImage = fullImage.to("cuda" if torch.cuda.is_available() else "cpu")#get a random index
            
        randomImage = randomImage.to("cuda" if torch.cuda.is_available() else "cpu")

        #toss a coin to decide whether to use the expanded image or a random real image
        coin = torch.rand(1)
        if coin < 0.5:
            image = randomImage
            #add random gaussian noise N(0, I)
            image = image + torch.normal(0, 1, size=image.shape).to("cuda" if torch.cuda.is_available() else "cpu")
            
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
                 contentLoss = mse(styleExtractor(fake), styleExtractor(fullImage))
            GFinalLoss = GLoss + 100 * L1Loss + 50 * contentLoss                    
 

        optGen.zero_grad()
        gScalar.scale(GFinalLoss).backward()
        gScalar.step(optGen)
        gScalar.update()

        

        
        writer.add_scalar("Generator Loss", GFinalLoss, global_step=step)
        writer.add_scalar("Discriminator Loss", DLoss, global_step=step)
        step += 1

        loop.set_postfix(Disc_Loss=DLoss.item(), Gen_Loss=GFinalLoss.item(), Epoch=epoch)
        """
        trainImage = torchvision.transforms.Pad(padding= (128, 128, 128, 128),padding_mode="constant")(trainImage)
        save_image(torch.cat((trainImage   * 0.5 + 0.5, fullImage   * 0.5 + 0.5, fake  * 0.5 + 0.5), dim=3), f"./tmp/water_test_{epoch}.png")
        """

   
    if epoch % 250 == 0:
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
        res = nn.Sequential(*list(res.children())[:-2])
        for param in res.parameters():
            param.requires_grad = False
        res.eval()
        print("=> Resnet loaded")
        #print(res)

        #print(res(torch.randn((1, 3, 256, 256)).to("cuda" if torch.cuda.is_available() else "cpu")).shape)
        #exit()

    dataPath = "data"

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

    elif dataName == "striped":
        checkpoints = config.weightsName.striped_SPECIALIZED
        print("=> Loading Striped")

    elif dataName == "timber":
        checkpoints = config.weightsName.timber_HighlySpecialized
        print("=> Loading Timber")
    elif dataName == "water":
        checkpoints = config.weightsName.water_HighlySpecialized
        print("=> Loading Water")

    elif dataName == "roofs":
        checkpoints = config.weightsName.roofs_SPECIALIZED
        dataPath = "data/Roofs"
        print("=> Loading Roofs")

    elif dataName == "grassWithRocks":
        checkpoints = config.weightsName.grassWithRocks_HighlySpecialized
        
        print("=> Loading Grass With Rocks")

    elif dataName == "grassWithRocks2":
        checkpoints = config.weightsName.grassWithRocks2_HighlySpecialized
        
        print("=> Loading Grass With Rocks 2: training also on random noise")

    elif dataName == "grass":
        checkpoints = config.weightsName.grass_HighlySpecialized

        print("=> Loading Grass")

    elif dataName == "stars":
        checkpoints = config.weightsName.stars_HighlySpecialized    
        print("=> Loading Stars")

    else:
        print("=> Invalid data name")
        exit()

    if loadModel:
        #check if the checkpoints exist
        if not os.path.exists(checkpoints["generator"]) or not os.path.exists(checkpoints["discriminator"]):
            print("=> Checkpoints not found")
            print(f"=> Model weights will be saved to {checkpoints['generator']} and {checkpoints['discriminator']}")
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



    dataset = Textures(dataPath = dataPath, trainingTarget=dataName, highResImagePath=checkpoints["highResImagePath"])
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
    
    fire.Fire(main)

  

   
