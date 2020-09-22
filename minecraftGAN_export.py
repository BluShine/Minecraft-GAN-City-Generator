from generatorGAN import makeStructures
from schematic import SchematicFile

import generatorGAN
import numpy as np

EXPORTGRID = 16
EXPORTSPACING = 2
BASE = 2
B2 = BASE * 2
B3 = BASE * 4

EXPORTPATH = "data/generatedExample.schematic"

def generateSampleStructures():
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = makeStructures(EXPORTGRID**2)

  #export a grid of generated structures.
  world = np.empty((B3, (B3 + EXPORTSPACING) * EXPORTGRID, 0), np.uint8)
  for i in range(EXPORTGRID):
      row = np.empty((B3, 0, B3), np.uint8)
      for j in range(EXPORTGRID):
          structure = predictions[i*EXPORTGRID + j, :, :, :]
          structure = np.concatenate((structure, 
                np.zeros((B3, EXPORTSPACING, B3), 
                dtype=np.uint8)), axis=1)
          row = np.concatenate((row, structure), axis=1)
      world = np.concatenate((world, row), axis=2)
      world = np.concatenate((world, 
                np.zeros((B3, (B3 + EXPORTSPACING) * EXPORTGRID, EXPORTSPACING), 
                dtype=np.uint8)), axis=2)
  return world

exportArea = generateSampleStructures()
print("generated %s" % str(exportArea.shape))
exportSchematic = SchematicFile(shape=exportArea.shape)
exportSchematic.blocks = exportArea
exportSchematic.save(EXPORTPATH)
print("exported to " + "data/generatedExample.schematic")