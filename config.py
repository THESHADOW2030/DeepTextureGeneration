class weightsName:
    GENERAL = {
            "generator" : "./weights/generalGeneratorWeights.pth.tar", 
            "discriminator" : "./weights/generalDiscriminatorWeights.pth.tar"
        }
    GENERAL_STYLE = {
            "generator" : "./weights/generalStyleGeneratorWeights.pth.tar",
            "discriminator" : "./weights/generalStyleDiscriminatorWeights.pth.tar"
        }
    
    
class trainingData:
    mode = {
            "full" : "full",
            "subset" : "subset"
    }


learningRate = 2e-4
batchSize = 8
numWorkers = 4