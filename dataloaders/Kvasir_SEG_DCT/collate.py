"""Face_FOMO_DCT batching plus the stacked per-block polyp mask coverage."""

import torch

from ..Face_FOMO_DCT.collate import face_fomo_dct_collate


def kvasir_seg_dct_collate(samples):
    batch = face_fomo_dct_collate(samples)
    batch["gt_mask"] = torch.stack([sample["gt_mask"] for sample in samples])
    return batch
