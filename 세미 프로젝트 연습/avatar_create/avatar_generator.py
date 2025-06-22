from PIL import Image, ImageEnhance

def generate_avatar(face_img: Image.Image) -> Image.Image:
    """
    얼굴 이미지를 입력받아 간단한 보정 및 필터를 적용한 아바타 이미지 생성
    (실제 환경에서는 AI 기반 아바타 생성 모델로 대체 가능)
    """
    # 밝기 조정
    enhancer = ImageEnhance.Brightness(face_img)
    avatar = enhancer.enhance(1.2)

    # 대비 조정
    enhancer = ImageEnhance.Contrast(avatar)
    avatar = enhancer.enhance(1.3)

    return avatar
