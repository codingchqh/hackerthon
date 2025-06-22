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
from avatar_create.avatar_generator import generate_avatar


from camera.face_capture import extract_face
from summarizer.gpt_summarizer import summarize_text, generate_video_script

# --- 플랫폼 확인 (로컬/클라우드 구분) ---
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    import sounddevice as sd

# --- 모델 로드 함수 (캐시 제거) ---
def load_model():
    return whisper.load_model("base")

# --- 세션 상태 초기화 ---
if "model" not in st.session_state:
    st.session_state.model = None

# --- UI 시작 ---
st.set_page_config(page_title="AI 아바타 + Whisper 전사", layout="centered")
st.title("📸 AI 아바타 생성 + 🎤 음성 전사 & 영상 생성")

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

# 탭으로 선택 (촬영 또는 업로드)
tab1, tab2 = st.tabs(["📸 카메라 촬영", "🖼️ 이미지 업로드"])

image_pil = None

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

# 얼굴 추출 및 아바타 생성
if image_pil:
    face_img = extract_face(image_pil)
    if face_img is None:
        st.error("😢 얼굴을 인식하지 못했습니다. 다른 이미지를 시도해주세요.")
    else:
        st.image(face_img, caption="✂️ 추출된 얼굴", width=256)

        # ✅ 아바타 생성
        avatar_img = generate_avatar(face_img)
        st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True)

        # ✅ 저장 경로 구성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"image/avatar_{timestamp}.jpg"
        avatar_img.save(save_path)
        st.success(f"✅ 아바타 이미지 저장 완료: {save_path}")

        # ✅ 세션 상태 저장
        st.session_state["saved_image_path"] = save_path

# --- 2️⃣ 이름/생년 기반 프롬프트 생성 ---
st.title("맞춤형 영상 프롬프트 생성기 🎬")

name = st.text_input("이름을 입력하세요")
birth_year = st.number_input("태어난 년도를 입력하세요", min_value=1900, max_value=datetime.now().year, step=1)

def get_age(birth_year):
    current_year = datetime.now().year
    return current_year - birth_year

if st.button("프롬프트 생성"):
    if not name:
        st.warning("이름을 입력해주세요.")
    else:
        age = get_age(birth_year)
        st.write(f"안녕하세요, {name}님! 현재 나이는 {age}세입니다.")

        if age < 20:
            prompt = f"{name}님의 어린 시절 모습을 담은 밝고 활기찬 영상"
        elif age < 40:
            prompt = f"{name}님의 젊고 역동적인 모습을 담은 세련된 영상"
        elif age < 60:
            prompt = f"{name}님의 성숙하고 안정된 모습을 담은 따뜻한 영상"
        else:
            prompt = f"{name}님의 인생의 지혜와 경험을 담은 감동적인 영상"

        st.info(prompt)
        st.session_state["video_prompt"] = prompt

# --- 3️⃣ 음성 녹음 및 Whisper 전사 ---

def record_audio(duration_sec=5, fs=16000, device=None):
    if not IS_LOCAL:
        st.error("⚠️ 로컬에서만 녹음이 가능합니다.")
        return None
    st.info(f"{duration_sec}초간 녹음 중...")
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    st.success("녹음 완료!")
    return audio.flatten()

def numpy_to_wav_bytes(audio_np, fs=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio(model, wav_io):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_io.read())
        tmp_path = tmp_file.name

    try:
        result = model.transcribe(tmp_path)
    finally:
        os.remove(tmp_path)

    transcript = result["text"]
    summary = summarize_text(transcript)
    script = generate_video_script(summary)
    return transcript, summary, script

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

uploaded_file = st.file_uploader("또는 오디오 파일(.wav/.mp3)을 업로드하세요", type=["wav", "mp3"])

if uploaded_file:
    if st.session_state.model is None:
        st.warning("먼저 모델을 로드해주세요!")
    else:
        st.audio(uploaded_file, format="audio/wav")
        transcript, summary, script = transcribe_audio(st.session_state.model, uploaded_file)
        st.subheader("📝 전사 결과")
        st.write(transcript)
        st.subheader("🔍 요약")
        st.write(summary)
        st.subheader("🎬 감성 영상 스크립트")
        st.write(script)
        st.session_state["script"] = script

# --- 4️⃣ 영상 생성 ---
st.header("4️⃣ 영상 생성")

prompt = st.session_state.get("video_prompt", None)
image_path = st.session_state.get("saved_image_path", None)
script = st.session_state.get("script", None)

def create_video_from_text_and_image(full_prompt, image_path):
    st.info(f"영상 생성 중...\n\n🧾 프롬프트: {full_prompt}\n🖼️ 이미지: {image_path}")
    st.success("✅ (예시) 영상 생성 완료!")

if prompt and image_path and os.path.exists(image_path):
    st.image(image_path, caption="🎨 최종 영상용 얼굴 이미지", use_container_width=True)

    if script:
        full_prompt = f"{prompt}\n\n🗣️ 감성 대사:\n{script}"
    else:
        full_prompt = prompt

    st.info(f"🧾 영상 프롬프트:\n\n{full_prompt}")

    if st.button("🎞️ 영상 만들기"):
        create_video_from_text_and_image(full_prompt, image_path)
else:
    st.warning("⚠️ 영상 생성을 위해 이름, 생년, 얼굴 사진, 음성 등을 모두 입력해주세요.")
