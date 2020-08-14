# tools for operating on schematics
import numpy as np
from schematic import SchematicFile

# naturally-occurring leaves, plants, liquids, cobwebs, etc.
TRANSPARENT = np.array([6,8,9,10,11,18,30,31,32,37,38,39,40,51,78,79,81,83,106,111,161,175], dtype=int)
# logs, pumpkin, melon, big mushroom.
SOLIDPLANTS = np.array([17,86,99,100,103,162], dtype=int)
# dirt, gravel, sand, grass, etc.
SOIL = np.array([2,3,12,13,82,80,110], dtype=int)

# stairs and half slabs
HALFBLOCK = np.array([44,53,67,108,109,114,126,128,134,135,136,156,163,164,180,182,203,205])
# built light-transmitting blocks
GLASS = np.array([20,95,102,160])
FENCE = np.array([85,107,113,183,184,185,186,187,188,189,190,191,192,
    139,101])
DECOR = np.array([177, 63,65,92,116,117,118,119,120,122,130,138,140,144,145,146,166,171,176,198,217,
    219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,255])
DOOR = np.array([64,71,193,194,195,196,197])
TRAPDOOR = np.array([96,167])
MACHINES = np.array([27,28,50,54,55,66,69,70,72,75,76,77,93,94,131,132,143,147,148,
    149,150,151,154,157,178,218])
CROPS = np.array([59,115,127,141,142,199,200,207])


DEFORESETER = np.concatenate((TRANSPARENT, SOLIDPLANTS))
ANNIHILATOR = np.concatenate((DEFORESETER, SOIL))
CLEARWALLS = np.concatenate((GLASS, FENCE)) 
MISC = np.concatenate((DECOR,DOOR,TRAPDOOR,MACHINES,CROPS))
NONFULLBLOCK = np.concatenate((HALFBLOCK, MISC, TRANSPARENT))

def loadArea(path:str) :
    area = SchematicFile.load(path)
    # use numpy for speed
    blockArr = np.array(area.blocks, dtype=int)
    # the arr can have negative entries for some reason. This fixes it.
    blockArr = np.mod(blockArr, 256)
    return blockArr

# removes all foliage and liquids from an area 
def deforest(area:np.ndarray) :
    return np.where(np.isin(area, DEFORESETER), 0, area)

def annihilate(area:np.ndarray) :
    return np.where(np.isin(area, ANNIHILATOR), 0, area)

def asBoolean(area:np.ndarray) :
    return np.ndarray.astype(area, dtype=bool)

def simplify(area:np.ndarray) :
    return asBoolean(np.where(np.isin(area, NONFULLBLOCK), 0, area))

def randomBlit(area:np.ndarray, mask:np.ndarray, sprite:np.ndarray) :
    # assume sprites are cubes.
    size = sprite.shape[0]
    y = np.random.randint(0, area.shape[0] - size)
    z = np.random.randint(0, area.shape[1] - size)
    x = np.random.randint(0, area.shape[2] - size)
    # slice the area, then overwrite air with our sprite.
    slicedArea = area[y:y+size,z:z+size,x:x+size]
    slicedmask = mask[y:y+size,z:z+size,x:x+size]
    slicedArea[slicedmask == 0] = sprite[slicedmask == 0]

def randomBlitSurface(area:np.ndarray, mask:np.ndarray, sprite:np.ndarray) :
    # assume sprites are cubes.
    size = sprite.shape[0]
    z = np.random.randint(0, area.shape[1] - size)
    x = np.random.randint(0, area.shape[2] - size)
    # find the surface by searching for the first nonzero index from the top.
    y = min(mask.shape[0], mask.shape[0] - np.argmax(np.flip(mask[0:,z,x], 0)))
    # print(y)
    # slice the area, then overwrite air with our sprite.
    slicedArea = area[y:y+size,z:z+size,x:x+size]
    slicedmask = mask[y:y+size,z:z+size,x:x+size]
    slicedArea[slicedmask == 0] = sprite[slicedmask == 0]

def randomSplat(area:np.ndarray, sprites:np.ndarray, splats:int) :
    mask = np.logical_not(simplify(area))
    for _ in range(splats) :
        randomBlit(area, mask, sprites[np.random.randint(0, sprites.shape[0])])

def randomSplatSurface(area:np.ndarray, sprites:np.ndarray, splats:int) :
    mask = simplify(area)
    for i in range(splats) :
        randomBlitSurface(area, mask, sprites[np.random.randint(0, sprites.shape[0])])
        if i % 100 == 0 : print(i)
    
