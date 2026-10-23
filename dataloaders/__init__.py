import copy
from .CoCo_Person.dataloader import CoCoPersonDataLoader
from .DareLab_Face.dataloader import DareLabFaceDataLoader
from .DareLab_Face_DCT.dataloader import DareLabFaceDCTDataLoader
from .Face_FOMO_DCT.dataloader import FaceFOMODCTDataLoader
from .Kvasir_SEG_DCT.dataloader import KvasirSEGDCTDataLoader


def build_dataloader(config, mode="train", pin_memory=None):
    dataloader_config = copy.deepcopy(config)
    name = dataloader_config.pop("name")

    if pin_memory is not None:
        dataloader_config["pin_memory"] = pin_memory

    
    if name in ("FullCoderDataset", "CoCoPersonDataLoader", "CoCo_Person"):
        dataloader = CoCoPersonDataLoader(**dataloader_config, mode=mode)
    elif name in ("DareLabFaceDataset", "DareLabFaceDataLoader", "DareLab_Face"):
        dataloader = DareLabFaceDataLoader(**dataloader_config, mode=mode)
    elif name in ("DareLabFaceDCTDataset", "DareLabFaceDCTDataLoader", "DareLab_Face_DCT"):
        dataloader = DareLabFaceDCTDataLoader(**dataloader_config, mode=mode)
    elif name in ("FaceFOMODCTDataset", "FaceFOMODCTDataLoader", "Face_FOMO_DCT"):
        dataloader = FaceFOMODCTDataLoader(**dataloader_config, mode=mode)
    elif name in ("KvasirSEGDCTDataset", "KvasirSEGDCTDataLoader", "Kvasir_SEG_DCT"):
        dataloader = KvasirSEGDCTDataLoader(**dataloader_config, mode=mode)
    else:
        raise NotImplementedError

    return dataloader
