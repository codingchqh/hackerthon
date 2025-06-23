# app.py (이전 코드와 동일)

import streamlit as st
from PIL import Image
import io
import numpy as np
import wave
import whisper
import tempfile
import os
import platform
from datetime import datetime

# avatar_generator 모듈에서 필요한 함수들을 임포트합니다.
from avatar_create.avatar_generator import generate_avatar_image, download_image_from_url

# camera/face_capture.py에 extract_face 함수가 있다고 가정합니다.
from camera.face_capture import extract_face

# summarizer/gpt_summarizer.py에 summarize_text, generate_video_script 함수가 있다고 가정합니다.
from summarizer.gpt_summarizer import summarize_text, generate_video_script

# --- 플랫폼 확인 (로컬/클라우드 구분) ---
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    import sounddevice as sd

# --- 모델 로드 함수 ---
def load_model():
    """Whisper 모델을 로드합니다."""
    return whisper.load_model("base")

# --- 세션 상태 초기화 ---
if "model" not in st.session_state:
    st.session_state.model = None
if "saved_image_path" not in st.session_state:
    st.session_state["saved_image_path"] = None
if "video_prompt" not in st.session_state:
    st.session_state["video_prompt"] = None
if "script" not in st.session_state:
    st.session_state["script"] = None

# --- UI 설정 ---
st.set_page_config(page_title="공감 on(溫)", layout="centered")
st.title("📸 공감 on(溫)")

# --- 0️⃣ 모델 로드 버튼 ---
if st.session_state.model is None:
    if st.button("모델 로드하기"):
        with st.spinner("모델을 로딩중입니다... 잠시만 기다려주세요."):
            st.session_state.model = load_model()
        st.success("✅ 모델이 성공적으로 로드되었습니다!")
else:
    st.info("✅ 모델이 로드되어 사용 가능합니다.")

# --- 디렉토리 생성 ---
os.makedirs("image", exist_ok=True)

# --- 1️⃣ 사진 입력: 카메라 촬영 또는 이미지 업로드 ---
st.header("1️⃣ 얼굴 이미지 입력 및 아바타 생성")

tab1, tab2 = st.tabs(["📸 카메라 촬영", "🖼️ 이미지 업로드"])
image_pil = None # 사용자가 선택한 원본 이미지 (PIL.Image.Image 객체)

with tab1:
    image_file = st.camera_input("아바타용 사진을 찍어보세요")
    if image_file:
        image_pil = Image.open(image_file)
        st.image(image_pil, caption="📷 촬영된 원본 이미지", use_container_width=True)

with tab2:
    uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="🖼️ 업로드된 이미지", use_container_width=True)

# 이미지가 입력되었을 때만 처리
if image_pil:
    face_img = None # 추출된 얼굴 이미지 (PIL.Image.Image 객체)
    try:
        face_img = extract_face(image_pil) # camera.face_capture 모듈의 함수 호출
    except Exception as e:
        st.error(f"얼굴 추출 중 오류 발생: {e}")
        face_img = None # 오류 발생 시 None으로 설정

    if face_img is None:
        st.error("😢 얼굴을 인식하지 못했습니다. 다른 이미지를 시도해주세요.")
    else:
        st.image(face_img, caption="✂️ 추출된 얼굴", width=256) # 추출된 얼굴 표시

        # 아바타 생성을 위한 프롬프트 설정 (텍스트 프롬프트)
        avatar_generation_prompt = "A friendly and expressive cartoon avatar, digital art style"

        avatar_urls = [] # generate_avatar_image가 반환할 URL 리스트
        try:
            # generate_avatar_image 함수 호출 (수정된 이름)
            # DALL-E는 텍스트 프롬프트만 받으므로, face_img가 아니라 텍스트 프롬프트를 전달합니다.
            avatar_urls = generate_avatar_image(prompt=avatar_generation_prompt, n_images=1)
        except Exception as e:
            st.error(f"아바타 이미지 생성 요청 중 오류 발생: {e}")
            avatar_urls = []

        avatar_img = None # 최종 아바타 이미지 (PIL.Image.Image 객체)
        if avatar_urls:
            print("생성된 아바타 이미지 URL:", avatar_urls[0])
            try:
                # 다운로드 함수 호출
                avatar_img = download_image_from_url(avatar_urls[0])
            except Exception as e:
                st.error(f"아바타 이미지 다운로드 또는 처리 중 오류 발생: {e}")
                avatar_img = None
        else:
            st.error("😢 아바타 이미지 URL 생성에 실패했습니다. OpenAI API 설정 또는 프롬프트를 확인해주세요.")
            avatar_img = None # URL이 없으면 이미지도 없음

        # avatar_img가 유효한 Image 객체일 때만 표시 및 저장
        if avatar_img is not None and isinstance(avatar_img, Image.Image):
            st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"image/avatar_{timestamp}.jpg"
            try:
                avatar_img.save(save_path)
                st.success(f"✅ 아바타 이미지 저장 완료: {save_path}")
                st.session_state["saved_image_path"] = save_path
            except Exception as e:
                st.error(f"아바타 이미지 저장 중 오류 발생: {e}")
        else:
            st.warning("아바타 이미지를 표시하거나 저장할 수 없습니다 (이미지 생성/다운로드 실패).")


# --- 2️⃣ 이름/생년 입력 및 프롬프트 생성 ---
st.title("맞춤형 영상 프롬프트 생성기 🎬")
name = st.text_input("이름을 입력하세요")
birth_year = st.number_input("태어난 년도를 입력하세요", min_value=1900, max_value=datetime.now().year, step=1)

def get_age(birth_year):
    """생년월일로 나이를 계산합니다."""
    return datetime.now().year - birth_year

if st.button("프롬프트 생성"):
    if not name:
        st.warning("이름을 입력해주세요.")
    else:
        age = get_age(birth_year)
        st.write(f"안녕하세요, {name}님! 현재 나이는 {age}세입니다.")
        # 나이에 따른 비디오 프롬프트 생성
        if age < 20:
            prompt = f"{name}님의 어린 시절 모습을 담은 밝고 활기찬 영상"
        elif age < 40:
            prompt = f"{name}님의 젊고 역동적인 모습을 담은 세련된 영상"
        elif age < 60:
            prompt = f"{name}님의 성숙하고 안정된 모습을 담은 따뜻한 영상"
        else:
            prompt = f"{name}님의 인생의 지혜와 경험을 담은 감동적인 영상"
        st.info(f"생성된 영상 프롬프트:\n\n{prompt}")
        st.session_state["video_prompt"] = prompt

# --- 3️⃣ 음성 녹음 및 Whisper 전사 ---

def record_audio(duration_sec=5, fs=16000, device=None):
    """마이크에서 오디오를 녹음합니다."""
    if not IS_LOCAL:
        st.error("⚠️ 로컬에서만 녹음이 가능합니다. 클라우드 환경에서는 오디오 파일 업로드를 이용해주세요.")
        return None
    
    if st.session_state.model is None:
        st.warning("Whisper 모델이 로드되지 않았습니다. 먼저 모델을 로드해주세요.")
        return None

    try:
        st.info(f"{duration_sec}초간 녹음 시작...")
        # sounddevice가 설치되어 있어야 합니다.
        audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
        sd.wait()
        st.success("녹음 완료!")
        return audio.flatten()
    except Exception as e:
        st.error(f"오디오 녹음 중 오류 발생: {e}")
        return None

def numpy_to_wav_bytes(audio_np, fs=16000):
    """Numpy 배열을 WAV 형식의 BytesIO 객체로 변환합니다."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit audio
        wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio(model, audio_input):
    """오디오를 Whisper 모델로 전사하고 요약 및 스크립트를 생성합니다."""
    tmp_path = None
    try:
        # Streamlit uploaded_file 객체는 BytesIO와 유사하게 작동합니다.
        # BytesIO나 업로드된 파일 객체를 임시 파일로 저장하여 Whisper 모델에 전달합니다.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            audio_input.seek(0) # 파일 포인터를 처음으로 이동 (중요!)
            tmp_file.write(audio_input.read()) # 오디오 입력의 내용을 읽어 임시 파일에 씁니다.
            tmp_path = tmp_file.name

        result = model.transcribe(tmp_path)
        transcript = result["text"]
        
        # summarizer 모듈의 함수 호출 (이 함수들이 정상 작동한다고 가정)
        summary = summarize_text(transcript) 
        script = generate_video_script(summary)
        
        return transcript, summary, script
    except Exception as e:
        st.error(f"오디오 전사/요약/스크립트 생성 중 오류 발생: {e}")
        return "", "", ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path) # 임시 파일 삭제

if IS_LOCAL and st.button("🎙 5초간 녹음하기"):
    if st.session_state.model is None:
        st.warning("먼저 모델을 로드해주세요!")
    else:
        audio_np = record_audio(duration_sec=5)
        if audio_np is not None:
            wav_bytes = numpy_to_wav_bytes(audio_np)
            st.audio(wav_bytes, format="audio/wav")
            transcript, summary, script = transcribe_audio(st.session_state.model, wav_bytes)
            
            st.subheader("📝 전사 결과")
            st.write(transcript)
            st.subheader("🔍 요약")
            st.write(summary)
            st.subheader("🎬 감성 영상 스크립트")
            st.write(script)
            st.session_state["script"] = script
        else:
            st.error("녹음된 오디오가 없습니다.")


uploaded_audio_file = st.file_uploader("또는 오디오 파일(.wav/.mp3)을 업로드하세요", type=["wav", "mp3"])
if uploaded_audio_file:
    if st.session_state.model is None:
        st.warning("먼저 모델을 로드해주세요!")
    else:
        st.audio(uploaded_audio_file, format=f"audio/{uploaded_audio_file.type.split('/')[-1]}")
        transcript, summary, script = transcribe_audio(st.session_state.model, uploaded_audio_file)
        
        st.subheader("📝 전사 결과")
        st.write(transcript)
        st.subheader("🔍 요약")
        st.write(summary)
        st.subheader("🎬 감성 영상 스크립트")
        st.write(script)
        st.session_state["script"] = script

# --- 4️⃣ 영상 생성 ---
st.header("4️⃣ 영상 생성")

# 세션 상태에서 필요한 정보 가져오기
prompt = st.session_state.get("video_prompt") # 2단계에서 생성된 영상 프롬프트
image_path = st.session_state.get("saved_image_path") # 1단계에서 저장된 아바타 이미지 경로
script = st.session_state.get("script") # 3단계에서 생성된 감성 스크립트

def create_video_from_text_and_image(full_prompt, image_path):
    """
    영상 생성 로직 (현재는 더미 함수)
    실제 구현 시 D-ID, RunwayML 등 외부 서비스 연동 필요
    """
    st.info(f"영상 생성 중...\n\n🧾 프롬프트: {full_prompt}\n🖼️ 이미지 경로: {image_path}")
    # 여기에 실제 영상 생성 API 호출 또는 로직을 구현합니다.
    # 예: D-ID API 호출, 이미지 및 텍스트를 전달하여 영상 생성
    st.success("✅ (예시) 영상 생성 완료! 실제 영상 생성 기능은 추후 연동됩니다.")


# 영상 생성 버튼 활성화 조건
can_create_video = False
if prompt and image_path and os.path.exists(image_path) and script:
    st.image(image_path, caption="🎨 최종 영상용 얼굴 이미지", use_container_width=True)
    full_prompt = f"{prompt}\n\n🗣️ 감성 대사:\n{script}"
    st.info(f"🧾 영상 프롬프트:\n\n{full_prompt}")
    can_create_video = True
else:
    st.warning("⚠️ 영상 생성을 위해 이름/생년 입력, 얼굴 사진, 음성 등을 모두 입력해주세요.")
    missing_items = []
    if not prompt: missing_items.append("영상 프롬프트")
    if not image_path or not os.path.exists(image_path): missing_items.append("아바타 이미지")
    if not script: missing_items.append("감성 스크립트 (음성 전사/요약)")
    if missing_items: # 누락된 항목이 있을 때만 표시
        st.info(f"누락된 항목: {', '.join(missing_items)}")


if can_create_video:
    if st.button("🎞️ 영상 만들기"):
        with st.spinner("영상을 생성 중입니다. 잠시만 기다려주세요..."):
            create_video_from_text_and_image(full_prompt, image_path)