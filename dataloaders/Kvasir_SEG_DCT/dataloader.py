"""DataLoader for Kvasir-SEG; arguments and batching are those of FaceFOMODCTDataLoader plus gt_mask."""

from ..Face_FOMO_DCT.dataloader import FaceFOMODCTDataLoader
from .collate import kvasir_seg_dct_collate
from .dataset import KvasirSEGDCTDataset


class KvasirSEGDCTDataLoader(FaceFOMODCTDataLoader):
    DATASET = KvasirSEGDCTDataset
    COLLATE = staticmethod(kvasir_seg_dct_collate)
