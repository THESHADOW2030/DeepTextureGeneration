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
                        transforms.Resize(width = 512, height = 512),
                        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5], max_pixel_value=255),
                        ToTensorV2()
                  ]),
                  trainingMode = "full"
        ):
        super(Dataset, self).__init__()

        self.path = dataPath
        self.data = os.listdir(self.path)
        self.trainingMode = trainingMode

        if self.trainingMode == "subset":
            #filter out the images that do not start with a number
            self.data = [image for image in self.data if image[0].isdigit()]

        self.transform = transform

        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        image  = Image.open(os.path.join(self.path, self.data[index % len(self.data)])).convert('RGB')
        image = np.array(image) 

        real = Image.open(os.path.join(self.path, self.data[np.random.randint(0, len(self.data))])).convert('RGB')
        real = np.array(real)

        if self.transform is not None:
            image = self.transform(image = image)
            image = image["image"].float()

            real = self.transform(image = real)
            real = real["image"].float()




        #crop the image at the center by half
        train = image[:, 256:768, 256:768]


        

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
    


    



            



