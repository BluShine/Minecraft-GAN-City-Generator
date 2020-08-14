# coding=UTF-8
""" Defines an nbtlib schema for schematic files """
from typing import Tuple
import enum
import numpy as np
import nbtlib as nbt


class Entity(nbt.CompoundSchema):
    """
    Entities describe objects which are not anchored to blocks, like mobs.
    """
    schema = {
        'id': nbt.String,
        'Pos': nbt.List[nbt.Double],
        'Motion': nbt.List[nbt.Double],
        'Rotation': nbt.List[nbt.Float],
        'FallDistance': nbt.Float,
        'Fire': nbt.Short,
        'Air': nbt.Short,
        'OnGround': nbt.Byte,
        'NoGravity': nbt.Byte,
        'Invulnerable': nbt.Byte,
        'PortalCooldown': nbt.Int,
        'UUIDMost': nbt.Long,
        'UUIDLeast': nbt.Long
    }


class BlockEntity(nbt.CompoundSchema):
    """
    Block entities contain additional metadata for placed blocks
    """
    schema = {
        'id': nbt.String,
        'x': nbt.Int,
        'y': nbt.Int,
        'z': nbt.Int
    }


class Schematic(nbt.CompoundSchema):
    """
    Schematic files represent a small section of a level

    Key fields include:

    - `Blocks`: A dense array of block IDs at each coordinate. Sorted by block
      height (bottop to top), then length (``Z``), then width (``X``). The
      index of the block at
      ``blocks[X, Y, Z]`` is ``Y * length * width + Z * width + X.``
    - `Data`: A dense array of data values for each block. This field shares
      sizes and indexing with `Blocks`.
    - `Entities`: A list of Compound tags which are entities.
    - `TileEntities`: A list of Compound tags which are block entities, which
      were previously known as tile entities.
    """
    schema = {
        'Height': nbt.Short,
        'Length': nbt.Short,
        'Width': nbt.Short,
        'Materials': nbt.String,
        'Blocks': nbt.ByteArray,
        'Data': nbt.ByteArray,
        'Entities': nbt.List[Entity],
        'TileEntities': nbt.List[BlockEntity]
    }


class SchematicFileRoot(nbt.CompoundSchema):
    """
    Describes the root element of a schematic file
    """
    schema = {
        'Schematic': Schematic
    }


class Material(enum.Enum):
    """
    Block Materials

    This enumeration indicates whether the block IDs in this schematic
    are to be taken from `Classic`, `Pocket`, or `Alpha` versions.
    Versions beyond `Alpha`—including `Beta` and stable builds—share a
    compatible set of block IDs. `Alpha` is the default for all
    newly-created schematics.
    """
    Classic = "Classic"
    Pocket = "Pocket"
    Alpha = "Alpha"


class SchematicFile(nbt.File, SchematicFileRoot):
    """
    Schematic File

    Schematic files are commonly used by world editors such as MCEdit,
    Schematica, and WorldEdit. They are intended to represent a small
    section of a level for the purposes of interchange or permanent
    storage.

    The origin of the schematic is always ``X = 0``, ``Y = 0``, ``Z = 0``.
    All positions for blocks, entities, and block entities are transformed
    into the schematic's coordinate system.

    Schematic coordinates map directly to data indices. Blocks and block
    data are stored in contiguous numpy byte arrays. The first dimension
    in these arrays is height (``Y``). The second and third dimensions
    are ``Z`` and ``X``, respectively.
    """

    def __init__(self, shape: Tuple[int, int, int] = (1, 1, 1),
                 blocks=None, data=None):
        super().__init__({'Schematic': {}})
        self.gzipped = True
        self.byteorder = 'big'
        self.root_name = 'Schematic'
        self.material = Material.Alpha
        self.resize(shape)
        if blocks is not None:
            self.blocks = blocks
        if data is not None:
            self.data = data
        self.entities = nbt.List()
        self.blockentities = nbt.List()

    def resize(self, shape: Tuple[int, int, int]) -> None:
        """
        Resize the schematic file

        Resizing the schematic clears the blocks and data

        :param shape: New dimensions for the schematic, as a tuple of
               ``(n_y, n_z, n_x)``.
        """

        self.root['Height'] = nbt.Short(shape[0])
        self.root['Length'] = nbt.Short(shape[1])
        self.root['Width'] = nbt.Short(shape[2])
        self.blocks = np.zeros(shape, dtype=np.uint8, order='C')
        self.data = np.zeros(shape, dtype=np.uint8, order='C')

    @classmethod
    def load(cls, filename, gzipped=True, byteorder='big') -> 'SchematicFile':
        """
        Load a schematic file from disk

        If the schematic file is already loaded into memory, use the
        :meth:`~from_buffer()` method instead.

        :param filename: Path to a schematic file on disk.
        :param gzipped: Schematic files are always stored gzipped. This option
               defaults to True
        :param byteorder: Schematic files are always stored in big endian
               number format.
        :return: Loaded schematic
        """
        return super().load(filename=filename,
                            gzipped=gzipped, byteorder=byteorder)

    @property
    def material(self) -> Material:
        """
        Block materials used by this schematic

        This enumeration indicates whether the block IDs in this schematic
        are to be taken from `Classic`, `Pocket`, or `Alpha` versions.
        Versions beyond `Alpha`—including `Beta` and stable builds—share a
        compatible set of block IDs. `Alpha` is the default for all
        newly-created schematics.

        :return: Enumerated Material type
        """
        return Material[self.root['Materials']]

    @material.setter
    def material(self, value: Material = Material.Alpha):
        self.root['Materials'] = value.value

    @property
    def shape(self) -> Tuple[nbt.Short, nbt.Short, nbt.Short]:
        """ Schematic shape

        :return: Shape of the schematic, as a tuple of ``Y``, ``Z``, and ``X``
                 size.
        """
        return self.root['Height'], self.root['Length'], self.root['Width']

    @property
    def blocks(self) -> np.array:
        """ Block IDs

        Entries in this array are the block ID at each coordinate of
        the schematic. This method returns an nbtlib type, but you may
        coerce it to a pure numpy array with ``numpy.asarray()``

        :return: 3D array which contains a view into the block IDs.
                 Array indices are in ``Y``, ``Z``, ``X`` order.
        """
        return self.root['Blocks'].reshape(self.shape, order='C').view()

    @blocks.setter
    def blocks(self, value):
        if not np.all(value.shape == self.shape):
            raise ValueError("Input shape %s does not match schematic shape %s"
                             % (value.shape, self.shape))

        self.root['Blocks'] = nbt.ByteArray(value.reshape(-1))

    @property
    def data(self) -> nbt.ByteArray:
        """ Block data

        Entries in this array are the block data values at each
        coordinate of the schematic. Only the lower four bits
        are used.  This method returns an nbtlib type, but you may
        coerce it to a pure numpy array with ``numpy.asarray()``

        :return: 3D array which contains a view into the block data.
                 Array indices are in ``Y``, ``Z``, ``X`` order.
        """
        return self.root['Data'].reshape(self.shape, order='C').view()

    @data.setter
    def data(self, value):
        if not np.all(value.shape == self.shape):
            raise ValueError("Input shape %s does not match schematic shape %s"
                             % (value.shape, self.shape))

        self.root['Data'] = nbt.ByteArray(value.reshape(-1))

    @property
    def entities(self) -> nbt.List[nbt.Compound]:
        """ Entities

        Each Entity in the schematic is a Compound tag. The schema only
        represents keys which are common to all Entities.

        :return: List of entities
        """
        return self.root['Entities']

    @entities.setter
    def entities(self, value: nbt.List[nbt.Compound]):
        self.root['Entities'] = value

    @property
    def blockentities(self) -> nbt.List[nbt.Compound]:
        """ Block Entities

        Block entities were previously known as "tile entities" and
        contain extended attributes for placed blocks. The schematic
        only enforces keys which are common to all entities, including
        a position and an ID.

        :return: List of block entities
        """
        return self.root['TileEntities']

    @blockentities.setter
    def blockentities(self, value: nbt.List[nbt.Compound]):
        self.root['TileEntities'] = value

    def __enter__(self):
        return self.root
