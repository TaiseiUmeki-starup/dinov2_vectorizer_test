import os
from glob import glob
from typing import Any, Callable, List, Optional
from .extended import ExtendedVisionDataset
from .decoders import DecoderType

class CustomImageFolder(ExtendedVisionDataset):
    """
    クラス分けなしの画像群を読む簡易データセット。
    画像は ImageDataDecoder で PIL 読み込み、ラベルは None。
    """
    def __init__(
        self,
        *,
        root: str,
        wildcard: str = "*.*",  # png/jpg 想定
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.ImageDataDecoder,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform, image_decoder_type=image_decoder_type, **kwargs)
        exts = (".png", ".jpg", ".jpeg")
        self._image_paths: List[str] = sorted(
            p for p in glob(os.path.join(root, wildcard)) if os.path.splitext(p)[1].lower() in exts
        )
        if not self._image_paths:
            raise RuntimeError(f'no images under "{root}" matching "{wildcard}"')

    def get_image_data(self, index: int) -> bytes:
        with open(self._image_paths[index], "rb") as f:
            return f.read()

    def get_target(self, index: int):
        return None

    def __len__(self) -> int:
        return len(self._image_paths)
