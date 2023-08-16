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
                        ToTensorV2()
                  ])
        ):
        super(Dataset, self).__init__()

        self.path = dataPath
        self.data = os.listdir(self.path)

        self.transform = transform

        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        image  = Image.open(os.path.join(self.path, self.data[index])).convert('RGB')
        image = np.array(image)

        if self.transform is not None:
            image = self.transform(image = image)
            image = image["image"]


        #crop the image at the center by half
        train = image[:, 256:768, 256:768]

        return train, image
    





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
    


    



            



