# app.py
import streamlit as st # Streamlit 라이브러리 임포트: 웹 애플리케이션을 쉽게 만들 수 있게 해줍니다.
from PIL import Image # Pillow (PIL) 라이브러리 임포트: 이미지 파일을 다루는 데 사용됩니다.
import io # 인메모리(RAM) 상에서 파일처럼 데이터를 다룰 수 있게 해주는 라이브러리입니다.
import sounddevice as sd # 사운드 장치를 제어하고 오디오를 녹음하거나 재생하는 데 사용됩니다.
import numpy as np # 숫자 계산, 특히 배열(행렬) 연산에 특화된 라이브러리입니다.
import wave # WAV 오디오 파일 형식을 읽고 쓰는 데 사용됩니다.
import whisper # OpenAI의 음성-텍스트 변환(STT) 모델인 Whisper를 사용하기 위한 라이브러리입니다.
import tempfile # 임시 파일을 안전하게 생성하고 관리하는 데 사용됩니다.
import os # 운영체제와 상호작용하는 기능을 제공합니다 (파일 경로, 디렉토리 생성 등).
import datetime # 날짜와 시간을 다루는 데 사용됩니다.
from camera.face_capture import extract_face  # 'camera' 패키지 안에 있는 'face_capture' 모듈에서 'extract_face' 함수를 가져옵니다. 이 함수는 이미지에서 얼굴을 잘라내는 역할을 합니다.

# Whisper 모델 캐시 로드
@st.cache_resource # Streamlit의 데코레이터: 이 함수가 반환하는 리소스(여기서는 Whisper 모델)를 한 번 로드하면 앱 실행 동안 메모리에 캐시하여 재사용합니다. 덕분에 페이지가 새로고침되어도 모델을 다시 불러올 필요가 없어 속도가 빨라집니다.
def load_model():
    """
    Whisper "base" 모델을 로드하여 반환합니다.
    이 함수는 @st.cache_resource 데코레이터 덕분에 한 번만 실행되고 결과가 캐시됩니다.
    """
    return whisper.load_model("base") # "base" 크기의 Whisper 모델을 로드합니다.

def record_audio(duration_sec=5, fs=16000, device=None):
    """
    지정된 시간(초) 동안 오디오를 녹음합니다.
    Args:
        duration_sec (int): 오디오를 녹음할 총 시간(초)입니다. 기본값은 5초입니다.
        fs (int): 샘플링 주파수(초당 샘플 수)입니다. 기본값은 16000Hz입니다.
        device (int, optional): 오디오를 녹음할 입력 장치의 번호입니다. None이면 기본 장치를 사용합니다.
    Returns:
        numpy.ndarray: 녹음된 오디오 데이터를 담고 있는 평탄화된(1차원) NumPy 배열입니다.
    """
    st.info(f"{duration_sec}초간 녹음 시작...") # Streamlit 앱에 정보 메시지를 표시합니다.
    # sounddevice의 sd.rec 함수를 사용하여 오디오 녹음을 시작합니다.
    # int(duration_sec * fs): 녹음할 총 샘플 수입니다.
    # samplerate=fs: 샘플링 주파수를 설정합니다.
    # channels=1: 모노(단일 채널)로 녹음합니다.
    # dtype='int16': 16비트 정수형으로 오디오 데이터를 저장합니다.
    # device=device: 사용할 오디오 입력 장치를 지정합니다.
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait() # 녹음이 완전히 끝날 때까지 기다립니다.
    st.success("녹음 완료!") # 녹음이 성공적으로 완료되었음을 알리는 메시지를 표시합니다.
    return audio.flatten() # 녹음된 오디오 배열을 1차원으로 평탄화하여 반환합니다.

def numpy_to_wav_bytes(audio_np, fs=16000):
    """
    NumPy 배열 형태의 오디오 데이터를 WAV 형식의 바이트 스트림(메모리 내 파일)으로 변환합니다.
    이는 실제 파일로 저장하지 않고 메모리 상에서 오디오 데이터를 다룰 때 유용합니다.
    Args:
        audio_np (numpy.ndarray): 오디오 데이터가 담긴 NumPy 배열입니다.
        fs (int): 오디오의 샘플링 주파수입니다. 기본값은 16000Hz입니다.
    Returns:
        io.BytesIO: WAV 형식의 오디오 데이터가 담긴 메모리 스트림 객체입니다.
    """
    buffer = io.BytesIO() # 오디오 데이터를 저장할 메모리 내 버퍼를 생성합니다.
    with wave.open(buffer, 'wb') as wf: # 'wave' 모듈을 사용하여 버퍼에 WAV 파일 형식으로 데이터를 씁니다.
        wf.setnchannels(1) # WAV 파일의 채널 수를 1(모노)로 설정합니다.
        wf.setsampwidth(2) # 샘플 너비를 2바이트(16비트)로 설정합니다.
        wf.setframerate(fs) # 프레임 레이트(샘플링 주파수)를 설정합니다.
        wf.writeframes(audio_np.tobytes()) # NumPy 배열을 바이트 시퀀스로 변환하여 WAV 프레임으로 기록합니다.
    buffer.seek(0) # 버퍼의 현재 위치를 맨 처음(0)으로 되돌립니다. 다른 함수가 이 버퍼에서 데이터를 읽을 수 있도록 준비하는 단계입니다.
    return buffer # WAV 바이트 스트림이 담긴 버퍼를 반환합니다.

def transcribe_audio(model, wav_io):
    """
    WAV 바이트 스트림으로부터 오디오를 Whisper 모델을 사용하여 텍스트로 전사(음성-텍스트 변환)합니다.
    Whisper 모델이 파일 경로를 직접 필요로 하므로, 임시 파일을 생성하여 사용합니다.
    Args:
        model (whisper.model): 미리 로드된 Whisper 모델 객체입니다.
        wav_io (io.BytesIO): WAV 형식의 오디오 데이터가 담긴 메모리 스트림입니다.
    Returns:
        str: 오디오로부터 전사된 텍스트 문자열입니다.
    """
    # 임시 파일을 생성합니다. Whisper 모델은 파일 경로를 인자로 받는 경우가 많기 때문에, 메모리 스트림을 임시 파일로 저장합니다.
    # delete=False: 파일 사용 후 바로 삭제하지 않고, 나중에 명시적으로 삭제할 것입니다.
    # suffix=".wav": 임시 파일의 확장자를 .wav로 지정합니다.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_io.read()) # 메모리 스트림(wav_io)의 모든 내용을 임시 파일에 씁니다.
        tmp_path = tmp_file.name # 생성된 임시 파일의 전체 경로를 저장합니다.

    result = model.transcribe(tmp_path) # Whisper 모델의 transcribe 함수를 호출하여 임시 파일의 오디오를 텍스트로 변환합니다.
    os.remove(tmp_path) # 전사 작업이 완료된 후, 생성된 임시 파일을 삭제합니다.
    return result["text"] # 전사 결과 딕셔너리에서 "text" 키에 해당하는 전사된 텍스트를 반환합니다.

# Streamlit 페이지 구성 설정
st.set_page_config(page_title="AI 아바타 + 음성 녹음 & 전사", layout="centered") # 웹 페이지의 제목을 설정하고, 컨텐츠가 중앙에 오도록 레이아웃을 설정합니다.
st.title("📸 AI 아바타 생성 + 🎤 음성 녹음 & Whisper 전사") # 앱의 메인 제목을 표시합니다.

# ---
## 1️⃣ 사진 촬영 및 얼굴 추출
# ---
st.header("1️⃣ 사진 촬영 및 얼굴 추출") # 첫 번째 섹션의 제목을 표시합니다.
image_file = st.camera_input("아바타용 사진을 찍어보세요") # Streamlit에서 제공하는 카메라 입력 위젯을 사용하여 사용자가 사진을 찍을 수 있게 합니다.

if image_file: # 사용자가 사진을 촬영하여 image_file 객체가 존재한다면 (사진이 업로드되었다면)
    image_pil = Image.open(image_file) # 업로드된 사진 파일을 PIL 이미지 객체로 엽니다.
    st.image(image_pil, caption="📷 촬영된 원본 이미지", use_container_width=True) # 촬영된 원본 이미지를 웹 앱에 표시합니다. 이미지 아래에 캡션을 추가하고, 컨테이너 너비에 맞게 조절합니다.

    face_img = extract_face(image_pil) # 'extract_face' 함수를 호출하여 원본 이미지에서 얼굴을 추출합니다.
    if face_img is None: # 얼굴 추출에 실패한 경우 (얼굴을 찾지 못한 경우)
        st.error("😢 얼굴을 인식하지 못했습니다. 다시 시도해주세요.") # 에러 메시지를 표시합니다.
    else: # 얼굴 추출에 성공한 경우
        st.image(face_img, caption="✂️ 얼굴 영역 추출", width=256) # 추출된 얼굴 이미지를 표시합니다. 캡션을 추가하고 너비를 256픽셀로 제한합니다.
        st.write("🎨 AI 아바타 생성 (임시 버전)") # "AI 아바타 생성 (임시 버전)"이라는 텍스트를 표시합니다.
        avatar_img = face_img # 현재는 추출된 얼굴 이미지를 그대로 AI 아바타 이미지로 사용합니다 (별도의 아바타 생성 로직은 아직 없음).
        st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True) # 생성된 AI 아바타 이미지를 표시합니다.

        # 얼굴 이미지 저장 코드 추가
        # 로컬 파일 시스템에 추출된 얼굴 이미지를 저장하는 부분입니다.
        save_dir = "C:/Users/user/Desktop/세미 프로젝트 연습/image" # 이미지를 저장할 로컬 디렉토리 경로를 지정합니다. (Windows 경로 표기법)
        os.makedirs(save_dir, exist_ok=True) # 지정된 디렉토리가 존재하지 않으면 생성합니다. 'exist_ok=True'는 이미 디렉토리가 있어도 오류를 발생시키지 않도록 합니다.

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 현재 날짜와 시간을 "년월일_시분초" 형식의 문자열로 만듭니다. (예: 20231027_153045)
        save_path = os.path.join(save_dir, f"face_{timestamp}.jpg") # 저장 디렉토리와 타임스탬프를 사용하여 최종 저장될 파일의 전체 경로를 생성합니다.
        avatar_img.save(save_path) # PIL 이미지 객체인 'avatar_img'를 지정된 경로에 JPEG 형식으로 저장합니다.

        st.success(f"얼굴 이미지가 저장되었습니다:\n{save_path}") # 이미지가 성공적으로 저장되었음을 사용자에게 알리고 저장 경로를 보여줍니다.

        # (원래 여기에 다운로드 버튼 코드가 있었지만, 제공된 코드에서는 제거되었습니다.)

# ---
## 2️⃣ 음성 녹음 및 Whisper 전사
# ---
st.header("2️⃣ 음성 녹음 및 Whisper 전사") # 두 번째 섹션의 제목을 표시합니다.
model = load_model() # 캐시된 Whisper 모델을 로드합니다. (최초 실행 시에만 실제로 로드되고 이후에는 캐시된 것을 사용)

if st.button("5초간 녹음하기"): # "5초간 녹음하기" 버튼이 클릭되면 다음 코드를 실행합니다.
    audio_np = record_audio(duration_sec=5) # 5초 동안 오디오를 녹음하고 NumPy 배열로 결과를 받습니다.
    wav_bytes = numpy_to_wav_bytes(audio_np) # 녹음된 NumPy 배열 오디오를 WAV 형식의 바이트 스트림으로 변환합니다.
    st.audio(wav_bytes, format="audio/wav")  # 변환된 WAV 바이트 스트림을 Streamlit의 오디오 위젯으로 재생하여 사용자에게 들려줍니다.
    text = transcribe_audio(model, wav_bytes) # 변환된 WAV 바이트 스트림을 Whisper 모델로 전사하여 텍스트를 얻습니다.
    st.subheader("📝 전사 결과") # "전사 결과"라는 서브헤더를 표시합니다.
    st.write(text) # Whisper 모델이 전사한 텍스트를 웹 앱에 표시합니다.