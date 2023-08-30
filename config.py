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

    grassWithRocks_HighlySpecialized = {
        "generator" : "./weights/grassWithRocksGeneratorWeights.pth.tar",
        "discriminator" : "./weights/grassWithRocksDiscriminatorWeights.pth.tar",
        "dataName" : "grassWithRocks",
        "highResImagePath" : "./highResData/grassWithRocks.jpg"
    }

    grassWithRocks2_HighlySpecialized = {
        "generator" : "./weights/grassWithRocks2GeneratorWeights.pth.tar",
        "discriminator" : "./weights/grassWithRocks2DiscriminatorWeights.pth.tar",
        "dataName" : "grassWithRocks2",
        "highResImagePath" : "./highResData/grassWithRocks.jpg"
    }

    grass_HighlySpecialized = {
        "generator" : "./weights/grassGeneratorWeights.pth.tar",
        "discriminator" : "./weights/grassDiscriminatorWeights.pth.tar",
        "dataName" : "grass",
        "highResImagePath" : "./highResData/grass.jpg"
    }

    stars_HighlySpecialized = {
        "generator" : "./weights/starsGeneratorWeights.pth.tar",
        "discriminator" : "./weights/starsDiscriminatorWeights.pth.tar",
        "dataName" : "stars",
        "highResImagePath" : "./highResData/stars.jpg"
    }
    
    

    
    


learningRate = 2e-4
batchSize = 8
numWorkers = 4