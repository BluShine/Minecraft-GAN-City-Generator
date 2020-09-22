
from generatorGAN import makeStructures
from minecraftGAN_export import generateStructures
from schematic import SchematicFile
import copy
import numpy as np
import SchematicTools
import generatorGAN

## Generate a settlement

blockArr = SchematicTools.loadArea('data/test_schematic.schematic')
print("loaded area: %d, %d, %d" % blockArr.shape)
height = blockArr.shape[0]
width = blockArr.shape[1]
length = blockArr.shape[2]
output = SchematicFile(shape=blockArr.shape)

print("generating structures")
#structures = np.load('data/np_samples.npy')
structures = makeStructures(1000)
splats = 1000
print("splatting %d" % splats)
SchematicTools.randomSplatSurface(blockArr, gen, splats)

output.blocks = blockArr

output.save("data/test_output.schematic")
print("output to data/test_output.schematic")
