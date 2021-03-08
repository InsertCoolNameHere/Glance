from enum import Enum

class Phase(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

if __name__ == "__main__":
    file="L1C_T11TLG_A016269_20180803T185257-3.tif"
    ext = "-3.tif"

    print(file.endswith(ext))