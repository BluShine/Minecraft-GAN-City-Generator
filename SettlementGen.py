
from schematic import SchematicFile
import copy
import numpy as np
import SchematicTools

blockArr = SchematicTools.loadArea('data/test_schematic.schematic')
print("loaded area: %d, %d, %d" % blockArr.shape)
height = blockArr.shape[0]
width = blockArr.shape[1]
length = blockArr.shape[2]
output = SchematicFile(shape=blockArr.shape)

# TODO: preserve block data for unmodified blocks.

# blockArr = SchematicTools.deforest(blockArr)
print("loading structures")
structures = np.load('data/np_samples.npy')
splats = 1000
print("splatting %d" % splats)
SchematicTools.randomSplatSurface(blockArr, structures, splats)

output.blocks = blockArr

output.save("data/test_output.schematic")
print("output to data/test_output.schematic")
