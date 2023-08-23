class weightsName:
    GENERAL = {
            "generator" : "./weights/generalGeneratorWeights.pth.tar", 
            "discriminator" : "./weights/generalDiscriminatorWeights.pth.tar",
            "dataName" : "general",
            "highResImagePath" : ""
        }
    GENERAL_STYLE = {
            "generator" : "./weights/generalStyleGeneratorWeights.pth.tar",
            "discriminator" : "./weights/generalStyleDiscriminatorWeights.pth.tar",
            "dataName" : "general",
            "highResImagePath" : ""
        }
    
    bubbly_SPECIALIZED= {
            "generator" : "./weights/bubblyGeneratorWeights.pth.tar",
            "discriminator" : "./weights/bubblyDiscriminatorWeights.pth.tar",
            "dataName" : "bubbly",
            "highResImagePath" : ""
        }
    fibrous_SPECIALIZED= {
            "generator" : "./weights/fibrousGeneratorWeights.pth.tar",
            "discriminator" : "./weights/fibrousDiscriminatorWeights.pth.tar",
            "dataName" : "fibrous",
            "highResImagePath" : ""
        }
    
    striped_SPECIALIZED= {
            "generator" : "./weights/stripedGeneratorWeights.pth.tar",
            "discriminator" : "./weights/stripedDiscriminatorWeights.pth.tar",
            "dataName" : "striped",
            "highResImagePath" : ""
        }

    timber_HighlySpecialized= {
            "generator" : "./weights/timberGeneratorWeights.pth.tar",
            "discriminator" : "./weights/timberDiscriminatorWeights.pth.tar",
            "dataName" : "timber",
            "highResImagePath" : "./highResData/timber.jpg"
    }

    roofs_SPECIALIZED= {    
            "generator" : "./weights/roofsGeneratorWeights.pth.tar",
            "discriminator" : "./weights/roofsDiscriminatorWeights.pth.tar",
            "dataName" : "roofs",
            "highResImagePath" : ""
        }
    water_HighlySpecialized = {
        "generator" : "./weights/waterGeneratorWeights.pth.tar",
        "discriminator" : "./weights/waterDiscriminatorWeights.pth.tar",
        "dataName" : "water",
        "highResImagePath" : "./highResData/water.jpg"
    }
    

    
    


learningRate = 2e-4
batchSize = 8
numWorkers = 4