import os
from shutil import copyfile
from shutil import rmtree
import random

def split():
    datafolder = 'data/'
    trainfolder = 'train/'
    valfolder = 'val/'
    valpercentage = .2
    if not os.path.exists(datafolder):
        print(datafolder,'does not exist.')
        return;
    if os.path.exists(trainfolder):
        try:
            rmtree(trainfolder, ignore_errors=True)
            os.makedirs(trainfolder)
        except OSError as e:
            print(e)
    if os.path.exists(valfolder):
        try:
            rmtree(valfolder, ignore_errors=True)
            os.makedirs(valfolder)
        except OSError as e:
            print(e)
    for root, dirs, files in os.walk(datafolder):
        for dir in dirs:
            datasubfolder = datafolder+dir+'/'
            trainsubfolder = trainfolder+dir+'/'
            valsubfolder = valfolder+dir+'/'
            os.makedirs(trainsubfolder)
            os.makedirs(valsubfolder)
            files = os.listdir(datasubfolder)
            random.shuffle(files)
            for i in range(int(len(files)*(1-valpercentage))):
                if not i%1000:
                    print('train ' + dir + ' ' + str(i))
                file = files.pop()
                copyfile(datasubfolder + file, trainsubfolder + file)
            for i in range(len(files)):
                if not i%1000:
                    print('val ' + dir + ' ' + str(i))
                file = files.pop()
                copyfile(datasubfolder + file, valsubfolder + file)
split()
