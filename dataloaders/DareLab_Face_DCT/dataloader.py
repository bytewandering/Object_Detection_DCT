"""DataLoader with the local DCT dataset, transform and collate implementation."""

from torch.utils.data import DataLoader
from .dataset import DareLabFaceDCTDataset
from .collate import darelab_face_dct_collate


class DareLabFaceDCTDataLoader(DataLoader):
    """Build the local dataset; mode sets default shuffle.

    Dataset/transform/collate types are fixed. Other options follow PyTorch.
    workers is a compatibility alias for num_workers. With batch_sampler,
    leave batch_size=1, shuffle=None and drop_last=False, as in PyTorch.
    """

    def __init__(self, img_path="/home/hzhao/ZHN/datasets/DareLab_Face_DCT", ann_path=None,
                 mode="train",
                 num_dct_coefficients=64, normalization_stats=None,
                 augmentation=None, rgb_path="/home/hzhao/ZHN/datasets/DareLab_Face",
                 *, batch_size=1, shuffle=None, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, generator=None,
                 prefetch_factor=None, persistent_workers=False,
                 pin_memory_device="", in_order=True, workers=None):
        if workers is not None:
            if num_workers != 0 and num_workers != workers:
                raise ValueError("workers and num_workers disagree")
            num_workers = workers
        dataset = DareLabFaceDCTDataset(
            img_path=img_path, ann_path=ann_path, mode=mode,
            num_dct_coefficients=num_dct_coefficients, normalization_stats=normalization_stats,
            augmentation=augmentation, rgb_path=rgb_path,
        )
        if not len(dataset):
            raise ValueError(f"{mode} dataset must contain images")
        if shuffle is None:
            shuffle = mode == "train" and sampler is None and batch_sampler is None
        super().__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
            collate_fn=darelab_face_dct_collate, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context, generator=generator,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device, in_order=in_order,
        )
