# app.py
import streamlit as st # Streamlit 라이브러리 임포트: 웹 애플리케이션 구축용
from PIL import Image # Pillow (PIL) 라이브러리 임포트: 이미지 처리에 사용
import io # 인메모리 파일 작업을 위한 라이브러리 (바이트 스트림 처리)
import sounddevice as sd # 사운드 장치 제어를 위한 라이브러리 (녹음, 재생 등)
import numpy as np # 숫자 계산, 특히 배열 처리를 위한 라이브러리
import wave # WAV 오디오 파일 읽기/쓰기를 위한 라이브러리
import whisper # 음성-텍스트 변환 (STT)을 위한 OpenAI의 Whisper 모델 라이브러리
import tempfile # 임시 파일 생성을 위한 라이브러리
import os # 운영체제 기능 (파일 및 경로 처리) 사용을 위한 os 임포트
import datetime # 날짜 및 시간 처리를 위한 라이브러리
from camera.face_capture import extract_face  # 'camera' 패키지에서 얼굴 자르기(추출) 함수 임포트

# Whisper 모델 캐시 로드 함수
@st.cache_resource # Streamlit의 캐시 기능: 모델을 한 번 로드하면 다시 로드하지 않고 재사용
def load_model():
    """
    Whisper "base" 모델을 로드하고 캐시합니다.
    Returns:
        whisper.model: 로드된 Whisper 모델 객체.
    """
    return whisper.load_model("base")

def record_audio(duration_sec=5, fs=16000, device=None):
    """
    지정된 시간 동안 오디오를 녹음하고 진행 상황을 Streamlit에 표시합니다.
    Args:
        duration_sec (int): 녹음할 시간 (초). 기본값은 5초.
        fs (int): 샘플링 주파수. 기본값은 16000 Hz.
        device (int, optional): 사용할 오디오 입력 장치 번호. None이면 기본 장치 사용.
    Returns:
        numpy.ndarray: 녹음된 오디오 데이터 (평탄화된 NumPy 배열).
    """
    st.info(f"{duration_sec}초간 녹음 시작...") # 정보 메시지 표시
    # 오디오 녹음: duration_sec * fs 만큼의 샘플, fs 샘플링 주파수, 1 채널, 16비트 정수형, 지정된 장치
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait() # 녹음이 완료될 때까지 대기
    st.success("녹음 완료!") # 성공 메시지 표시
    return audio.flatten() # 녹음된 오디오 데이터를 1차원 배열로 평탄화하여 반환

def numpy_to_wav_bytes(audio_np, fs=16000):
    """
    NumPy 배열 형태의 오디오 데이터를 WAV 형식의 바이트 스트림으로 변환합니다.
    Args:
        audio_np (numpy.ndarray): 오디오 데이터가 담긴 NumPy 배열.
        fs (int): 샘플링 주파수. 기본값은 16000 Hz.
    Returns:
        io.BytesIO: WAV 형식의 오디오 데이터가 담긴 메모리 스트림.
    """
    buffer = io.BytesIO() # 데이터를 저장할 인메모리 바이트 버퍼 생성
    with wave.open(buffer, 'wb') as wf: # wave 모듈을 사용하여 버퍼에 WAV 파일 형식으로 쓰기
        wf.setnchannels(1) # 채널 수 설정 (모노)
        wf.setsampwidth(2) # 샘플 너비 설정 (2바이트 = 16비트)
        wf.setframerate(fs) # 프레임 레이트 (샘플링 주파수) 설정
        wf.writeframes(audio_np.tobytes()) # NumPy 배열을 바이트로 변환하여 WAV 프레임으로 기록
    buffer.seek(0) # 버퍼의 읽기/쓰기 위치를 처음으로 되돌림
    return buffer

def transcribe_audio(model, wav_io):
    """
    WAV 바이트 스트림을 Whisper 모델로 전사(transcribe)합니다.
    Args:
        model (whisper.model): 로드된 Whisper 모델 객체.
        wav_io (io.BytesIO): WAV 형식의 오디오 데이터가 담긴 메모리 스트림.
    Returns:
        str: 전사된 텍스트.
    """
    # 임시 WAV 파일 생성 (Whisper 모델이 파일 경로를 인자로 받기 때문)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_io.read()) # 메모리 스트림의 내용을 임시 파일에 기록
        tmp_path = tmp_file.name # 임시 파일의 경로 저장

    result = model.transcribe(tmp_path) # Whisper 모델을 사용하여 오디오를 텍스트로 전사
    os.remove(tmp_path) # 사용이 끝난 임시 파일 삭제
    return result["text"] # 전사된 텍스트 반환

# Streamlit 페이지 구성 설정
st.set_page_config(page_title="AI 아바타 + 음성 녹음 & 전사", layout="centered") # 페이지 제목 및 레이아웃 설정
st.title("📸 AI 아바타 생성 + 🎤 음성 녹음 & Whisper 전사") # 앱의 메인 제목

# 1️⃣ 사진 촬영 및 얼굴 추출 섹션
st.header("1️⃣ 사진 촬영 및 얼굴 추출")
image_file = st.camera_input("아바타용 사진을 찍어보세요") # Streamlit의 카메라 입력 위젯 (사진 촬영)

if image_file: # 사진이 촬영되었으면
    image_pil = Image.open(image_file) # 업로드된 사진 파일을 PIL 이미지 객체로 열기
    st.image(image_pil, caption="📷 촬영된 원본 이미지", use_container_width=True) # 원본 이미지 표시

    face_img = extract_face(image_pil) # 촬영된 이미지에서 얼굴 추출 함수 호출
    if face_img is None: # 얼굴이 인식되지 않은 경우
        st.error("😢 얼굴을 인식하지 못했습니다. 다시 시도해주세요.") # 에러 메시지 표시
    else: # 얼굴이 인식된 경우
        st.image(face_img, caption="✂️ 얼굴 영역 추출", width=256) # 추출된 얼굴 이미지 표시
        st.write("🎨 AI 아바타 생성 (임시 버전)") # AI 아바타 생성 (임시) 메시지
        avatar_img = face_img # 추출된 얼굴을 아바타 이미지로 사용 (현재는 단순 복사)
        st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True) # 아바타 이미지 표시

        # 추출된 얼굴 이미지 저장 코드 추가
        save_dir = r"C:/Users/user/Desktop/세미 프로젝트 연습/image" # 이미지를 저장할 디렉토리 경로
        os.makedirs(save_dir, exist_ok=True) # 디렉토리가 없으면 생성 (이미 존재해도 오류 발생 안함)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 시간을 기반으로 타임스탬프 생성
        save_path = os.path.join(save_dir, f"face_{timestamp}.jpg") # 저장할 파일의 전체 경로 생성
        avatar_img.save(save_path) # 아바타 이미지를 지정된 경로에 JPEG 형식으로 저장

        st.success(f"얼굴 이미지가 저장되었습니다:\n{save_path}") # 저장 성공 메시지 표시

        # 아바타 이미지 다운로드 버튼
        buf = io.BytesIO() # 다운로드를 위한 인메모리 바이트 버퍼 생성
        avatar_img.save(buf, format="JPEG") # 아바타 이미지를 JPEG 형식으로 버퍼에 저장
        buf.seek(0) # 버퍼의 읽기/쓰기 위치를 처음으로 되돌림
        st.download_button("📥 아바타 저장", data=buf, file_name="ai_avatar.jpg", mime="image/jpeg") # 다운로드 버튼 생성

# 2️⃣ 음성 녹음 및 Whisper 전사 섹션
st.header("2️⃣ 음성 녹음 및 Whisper 전사")
model = load_model() # Whisper 모델 로드 (캐시 사용)

if st.button("5초간 녹음하기"): # "5초간 녹음하기" 버튼이 클릭되면
    audio_np = record_audio(duration_sec=5) # 5초간 오디오 녹음
    wav_bytes = numpy_to_wav_bytes(audio_np) # 녹음된 NumPy 배열을 WAV 바이트 스트림으로 변환
    st.audio(wav_bytes, format="audio/wav")  # 녹음된 음성