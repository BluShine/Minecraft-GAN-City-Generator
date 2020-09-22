
from generatorGAN import makeStructures
from schematic import SchematicFile
import numpy as np
import SchematicTools
import generatorGAN

## Generate a settlement

INPUTWORLD = "data/example_world.schematic"
EXPORTPATH = "data/example_world_output.schematic"
BUILDINGSTOGENERATE = 10000
BUILDINGSTOSPLAT = 10000

blockArr = SchematicTools.loadArea(INPUTWORLD)
print("loaded area: %d, %d, %d" % blockArr.shape)
height = blockArr.shape[0]
width = blockArr.shape[1]
length = blockArr.shape[2]
output = SchematicFile(shape=blockArr.shape)

print("generating structures")
#structures = np.load('data/np_samples.npy')
structures = makeStructures(BUILDINGSTOGENERATE)
print("generated %s" % str(structures.shape))
print("splatting %d" % BUILDINGSTOSPLAT)
SchematicTools.randomSplatSurface(blockArr, structures, BUILDINGSTOSPLAT)

output.blocks = blockArr

output.save(EXPORTPATH)
print("output to %s" % EXPORTPATH)
