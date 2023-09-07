from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import albumentations as transforms
from albumentations.pytorch import ToTensorV2



class Textures(Dataset):
    def __init__(self,
                  dataPath, 
                  transform=transforms.Compose([
                        transforms.Resize(height = 1024, width = 1024),
                        transforms.RandomCrop(height = 512, width = 512),
                        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value=255),
                        ToTensorV2()
                  ]),
                  trainingTarget = "general",
                  highResImagePath = "" #it's a file
        ):
        super(Dataset, self).__init__()

        self.path = dataPath
        self.data = os.listdir(self.path)
        self.trainingTarget = trainingTarget
        self.highResImagePath = highResImagePath
        self.transform = transform
        self.highResImage = None

        #filter out the directories
        self.data = [image for image in self.data if os.path.isfile(os.path.join(self.path, image))]





        if self.trainingTarget != "general" and self.path == "data":
            #filter out the images that do not start with a number
            self.data = [image for image in self.data if image.startswith(self.trainingTarget)]
            



        if highResImagePath != "":
            self.data = [self.highResImagePath for _ in range(40)]
            self.highResImage = Image.open(self.highResImagePath).convert('RGB')
            self.highResImage = np.array(self.highResImage)

            trans = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5]), 
                transforms.Resize(height = 2048, width = 2048)
            ])
            self.highResImage = trans(image = self.highResImage)["image"]
    
            
            
        


        

    def __len__(self):
        return len(self.data)
    



    def __getitem__(self, index):
        """
        train: the image that is used to train the generator
        image: the full image of train
        real: a random image from the dataset
        """


        if self.highResImage is not None:

            image = self.highResImage

            real = self.highResImage

            trans = transforms.Compose([
                transforms.RandomCrop(width=512, height=512),
                ToTensorV2()
            ])

            image = trans(image = image)["image"]
            X, Y = image.shape[1] // 2, image.shape[2] // 2

            train = image[:, X - 128 : X + 128, Y - 128 : Y + 128]

            real = trans(image = real)["image"]

        

            

            return train.float(), image.float(), real.float()
            
            

        
        image  = Image.open(os.path.join(self.path, self.data[index % len(self.data)])).convert('RGB')
        image = np.array(image) 

        real = Image.open(os.path.join(self.path, self.data[np.random.randint(0, len(self.data))])).convert('RGB')
        real = np.array(real)

        if self.transform is not None:
            image = self.transform(image = image)

            image = image["image"].float()

            real = self.transform(image = real)
            real = real["image"].float()



        #crop image at the center point
        X, Y = image.shape[1] // 2, image.shape[2] // 2

        train = image[:, X - 128 : X + 128, Y - 128 : Y + 128]




        

        return train, image, real
    





if __name__ =="__main__":

    #show some images
    import torchvision.transforms as transforms

 

    dataset = Textures(dataPath = "data")

    #get random image
    image, test  = dataset[np.random.randint(0, len(dataset))]
    print(image.shape)
    print(test.shape)

    #show image
    import matplotlib.pyplot as plt
    #plot them on the same line
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image.permute(1,2,0))
    axarr[1].imshow(test.permute(1,2,0))
    plt.show()
    


    



            



