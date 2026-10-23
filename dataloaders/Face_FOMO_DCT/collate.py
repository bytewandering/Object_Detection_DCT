"""Samples share the DareLab DCT layout, so batching is identical."""

from ..DareLab_Face_DCT.collate import darelab_face_dct_collate as face_fomo_dct_collate

__all__ = ["face_fomo_dct_collate"]
