class weightsName:
    GENERAL = {
            "generator" : "./weights/generalGeneratorWeights.pth.tar", 
            "discriminator" : "./weights/generalDiscriminatorWeights.pth.tar",
            "dataName" : "general"
        }
    GENERAL_STYLE = {
            "generator" : "./weights/generalStyleGeneratorWeights.pth.tar",
            "discriminator" : "./weights/generalStyleDiscriminatorWeights.pth.tar",
            "dataName" : "general"
        }
    
    bubbly_SPECIALIZED= {
            "generator" : "./weights/bubblyGeneratorWeights.pth.tar",
            "discriminator" : "./weights/bubblyDiscriminatorWeights.pth.tar",
            "dataName" : "bubbly" 
        }
    fibrous_SPECIALIZED= {
            "generator" : "./weights/fibrousGeneratorWeights.pth.tar",
            "discriminator" : "./weights/fibrousDiscriminatorWeights.pth.tar",
            "dataName" : "fibrous"
        }
    
    striped_SPECIALIZED= {
            "generator" : "./weights/stripedGeneratorWeights.pth.tar",
            "discriminator" : "./weights/stripedDiscriminatorWeights.pth.tar",
            "dataName" : "striped"
        }
    

    
    


learningRate = 2e-4
batchSize = 8
numWorkers = 4