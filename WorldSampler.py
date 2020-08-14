from typing import ForwardRef
import SchematicTools
from schematic import SchematicFile
import numpy as np
import typing
import SchematicTools
from PIL import Image
import glob

SAMPLESIZE = 8
FILEPATH = 'data/np_samples_%dx.npy' % SAMPLESIZE
#worlds = glob.glob(SOURCEFOLDER + '/*.schematic')
WORLDS = ['D:/Projects/MinecraftMLGenerator/Data/zearth_160_272_41.schematic']
SAMPLECOUNT = 1000

def sample(area:np.ndarray, samples:int, size:int) :
    samplerY = np.random.randint(0, area.shape[0] - size, samples)
    samplerZ = np.random.randint(0, area.shape[1] - size, samples) 
    samplerX = np.random.randint(0, area.shape[2] - size, samples)
    sampler = np.stack((samplerY, samplerZ, samplerX), axis=-1)
    slices = np.empty((samples, size, size, size), dtype=int)
    for i in range(samples) :
        slices[i] = area[
            sampler[i,0]:sampler[i,0]+size, 
            sampler[i,1]:sampler[i,1]+size, 
            sampler[i,2]:sampler[i,2]+size]
    print("sampled %s" % str(slices.shape))
    return slices

def sampleFlat(area:np.ndarray, samples:int, size:int) :
    print("sampling flat %d, size %d" % (samples, size))
    samplerY = np.random.randint(0, area.shape[0], samples)
    samplerZ = np.random.randint(0, area.shape[1] - size, samples) 
    samplerX = np.random.randint(0, area.shape[2] - size, samples)
    sampler = np.stack((samplerY, samplerZ, samplerX), axis=-1)
    slices = np.empty((samples, size, size), dtype=int)
    for i in range(samples) :
        slices[i] = area[
            sampler[i,0], 
            sampler[i,1]:sampler[i,1]+size, 
            sampler[i,2]:sampler[i,2]+size]
    return slices

def filter(slices:np.ndarray, minFill, maxFill) :
    minBlocks = minFill * SAMPLESIZE**3
    maxBlocks = maxFill * SAMPLESIZE**3
    mask = np.logical_and(slices.sum(axis=(1,2,3)) > minBlocks, slices.sum(axis=(1,2,3)) < maxBlocks) 
    return slices[mask]

def filterFlat(slices:np.ndarray, threshold) :
    mask = slices.sum(axis=(1,2)) > threshold
    return slices[mask]

def showSamples(samples, width, height) :
    preview = np.empty((0, SAMPLESIZE * width,3), np.uint8)
    for i in range(width):
        row = np.empty((SAMPLESIZE, 0, 3), np.uint8)
        for j in range(height):
            imageR = samples[i*width + j, :, :, :] * 255
            imageG = np.average(imageR, axis=0)
            image = np.stack((imageR[0], imageG, imageG), axis=2)
            row = np.hstack((row, image))
        preview = np.vstack((preview, row))

    outputimage = Image.fromarray(preview.astype(np.uint8))
    outputimage.save('samplesPreview.png')
    outputimage.show()

def exportSamples(samples, width, height, spacing) :
    exportWorld = np.empty((SAMPLESIZE, (SAMPLESIZE + spacing) * width, 0), np.uint8)
    for i in range(width):
        row = np.empty((SAMPLESIZE, 0, SAMPLESIZE), np.uint8)
        for j in range(height):
            structure = samples[i*width + j, :, :, :] # load a sample
            #apply spacing
            structure = np.concatenate((structure, np.zeros((SAMPLESIZE, spacing, SAMPLESIZE), dtype=np.uint8)), axis=1)
            #add to the row
            row = np.concatenate((row, structure), axis=1)
        #add to the column
        exportWorld = np.concatenate((exportWorld, row), axis=2)
        #apply spacing
        exportWorld = np.concatenate((exportWorld, np.zeros((SAMPLESIZE, (SAMPLESIZE + spacing) * width, spacing), dtype=np.uint8)), axis=2)
    exportSchematic = SchematicFile(shape=exportWorld.shape)
    exportSchematic.blocks = exportWorld
    exportSchematic.save("data/sampledExample.schematic")


simpleWorlds = []
for w in WORLDS :
    simpleWorlds.append(SchematicTools.simplify(SchematicTools.loadArea(w)))
print("loaded %d worlds" % len(simpleWorlds))
samples = np.empty((0, SAMPLESIZE, SAMPLESIZE, SAMPLESIZE))
for s in simpleWorlds :
    samples = np.concatenate((samples, sample(s, SAMPLECOUNT, SAMPLESIZE)), axis=0)
#sampleWorld = SchematicTools.loadArea('data/sample_world.schematic')
#simplifiedWorld = SchematicTools.simplify(sampleWorld)
#print("loaded area: %d, %d, %d" % simplifiedWorld.shape)
#testSamples = sample(simplifiedWorld, SAMPLECOUNT, SAMPLESIZE)
print("sampled total %s" % str(samples.shape))
filtered = filter(samples, .05, .7)
print("filtered to %s" % str(filtered.shape))
np.save(FILEPATH, filtered)
print("saved to: %s" % FILEPATH)
#showSamples(filtered, 10, 10)
exportSamples(filtered, 16, 16, 2)