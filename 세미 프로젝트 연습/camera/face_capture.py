import cv2 # OpenCV 라이브러리 임포트: 이미지 및 비디오 처리에 사용
import numpy as np # NumPy 라이브러리 임포트: 숫자 배열 처리에 사용
from PIL import Image # PIL (Pillow) 라이브러리 임포트: 이미지 파일 처리에 사용
from typing import Optional # 타입 힌트 사용을 위한 Optional 임포트
import os # 운영체제 기능 (파일 및 경로 처리) 사용을 위한 os 임포트

def extract_face(image_pil: Image.Image, save_path: Optional[str] = None) -> Optional[Image.Image]:
    """
    입력된 PIL 이미지에서 가장 큰 얼굴을 찾아 잘라서 PIL 이미지로 반환합니다.
    얼굴이 없으면 None 반환.
    save_path가 주어지면 해당 경로에 얼굴 이미지를 저장합니다.
    """
    # PIL 이미지를 NumPy 배열로 변환하고 RGB 형식으로 변경 (OpenCV는 보통 BGR을 사용하지만, 여기서는 RGB로 변환하여 처리)
    img = np.asarray(image_pil.convert("RGB"))
    # RGB 이미지를 회색조로 변환 (얼굴 인식을 위해 일반적으로 회색조 이미지를 사용)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Haar Cascade 분류기 로드: 미리 훈련된 얼굴 감지 모델을 사용
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # 회색조 이미지에서 얼굴을 감지
    # scaleFactor: 이미지 스케일 감소 인자 (1.1은 이미지를 10%씩 줄여나가면서 얼굴을 찾음)
    # minNeighbors: 얼굴 후보 영역 주변의 이웃 개수 (높을수록 오탐 감소)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 감지된 얼굴이 없으면 None 반환
    if faces is None or len(faces) == 0:
        return None

    # 감지된 얼굴들 중에서 가장 큰 얼굴 (면적이 가장 큰 얼굴)을 선택
    # lambda rect: rect[2] * rect[3]는 각 얼굴 영역 (x, y, 너비, 높이)에서 너비*높이 (면적)를 계산하여 정렬 기준으로 사용
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # 원본 이미지의 높이와 너비를 가져옴
    height, width = img.shape[:2]
    # 얼굴 영역이 이미지 경계를 벗어나지 않도록 조정
    x, y = max(0, x), max(0, y) # x, y 좌표가 0보다 작으면 0으로 조정
    w = min(w, width - x) # 너비가 이미지 경계를 벗어나면 조정
    h = min(h, height - y) # 높이가 이미지 경계를 벗어나면 조정

    # 조정된 좌표를 사용하여 원본 이미지에서 얼굴 부분만 잘라냄
    face_img = img[y:y+h, x:x+w]
    # 잘라낸 얼굴 이미지를 PIL 이미지 객체로 변환
    face_img_pil = Image.fromarray(face_img)

    # save_path가 제공된 경우, 얼굴 이미지를 해당 경로에 저장
    if save_path:
        # 저장 경로의 디렉토리 부분을 가져옴
        save_dir = os.path.dirname(save_path)
        # 디렉토리가 존재하지 않으면 생성
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # PIL 이미지 객체를 지정된 경로에 저장
        face_img_pil.save(save_path)

    # 잘라낸 얼굴 PIL 이미지 객체를 반환
    return face_img_pil