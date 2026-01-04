import logging
import os
from glob import glob
from typing import Any, Callable, List, Optional

from .extended import ExtendedVisionDataset
from .decoders import DecoderType

logger = logging.getLogger("dinov2")


class MaskedPNG(ExtendedVisionDataset):
    """
    Simple dataset to load PNG images from a directory for self-supervised training.
    Labels are dummy (None); reading/transform is delegated to upstream transforms.
    """

    def __init__(
        self,
        *,
        root: str,
        wildcard: str = "*.png",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.ImageDataDecoder,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            image_decoder_type=image_decoder_type,
            **kwargs,
        )
        self.root = root
        self.wildcard = wildcard
        self._image_paths: List[str] = sorted(glob(os.path.join(root, wildcard)))

        if not self._image_paths:
            raise RuntimeError(f'no images found under "{root}" matching "{wildcard}"')

        logger.info(f"MaskedPNG: found {len(self._image_paths):,d} images")

    def get_image_data(self, index: int) -> bytes:
        image_full_path = self._image_paths[index]
        with open(image_full_path, "rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int):
        return None

    def __len__(self) -> int:
        return len(self._image_paths)
