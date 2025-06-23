import numpy as np
from PIL import Image
from typing import Optional
import os

def extract_face(image_pil: Image.Image, save_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    원본 PIL 이미지를 그대로 반환합니다.
    save_path가 주어지면 해당 경로에 이미지를 저장합니다.
    """
    # 원본 이미지를 그대로 사용
    face_img_pil = image_pil

    # save_path가 제공된 경우, 이미지를 해당 경로에 저장
    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        face_img_pil.save(save_path)

    return face_img_pil
