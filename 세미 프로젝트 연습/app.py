import streamlit as st
from PIL import Image
import io
import numpy as np
import wave
import whisper
import tempfile
import os
import datetime
import platform
from datetime import datetime

from camera.face_capture import extract_face
from summarizer.gpt_summarizer import summarize_text, generate_video_script

# --- 플랫폼 확인 (로컬/클라우드 구분) ---
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    import sounddevice as sd

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

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

    result = model.transcribe(tmp_path)
    os.remove(tmp_path)

    transcript = result["text"]
    summary = summarize_text(transcript)
    script = generate_video_script(summary)
    return transcript, summary, script

def get_age(birth_year):
    current_year = datetime.now().year
    return current_year - birth_year

def create_video_from_text_and_image(full_prompt, image_path):
    st.info(f"영상 생성 중...\n\n🧾 프롬프트: {full_prompt}\n🖼️ 이미지: {image_path}")
    st.success("✅ (예시) 영상 생성 완료!")

# --- UI 시작 ---
st.set_page_config(page_title="AI 아바타 + Whisper 전사", layout="centered")
st.title("📸 AI 아바타 생성 + 🎤 음성 전사 & 영상 생성")

# --- 1️⃣ 사진 촬영 ---
st.header("1️⃣ 사진 촬영 및 얼굴 추출")
image_file = st.camera_input("아바타용 사진을 찍어보세요")

if image_file:
    image_pil = Image.open(image_file)
    st.image(image_pil, caption="📷 촬영된 원본 이미지", use_container_width=True)

    face_img = extract_face(image_pil)
    if face_img is None:
        st.error("😢 얼굴을 인식하지 못했습니다.")
    else:
        st.image(face_img, caption="✂️ 얼굴 추출", width=256)
        avatar_img = face_img
        st.image(avatar_img, caption="🖼️ AI 아바타", use_container_width=True)

        save_dir = "image"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"face_{timestamp}.jpg")
        avatar_img.save(save_path)
        st.success(f"이미지 저장 완료: {save_path}")
        st.session_state["saved_image_path"] = save_path

# --- 2️⃣ 이름/생년 기반 프롬프트 생성 ---
st.title("맞춤형 영상 프롬프트 생성기 🎬")

name = st.text_input("이름을 입력하세요")
birth_year = st.number_input("태어난 년도를 입력하세요", min_value=1900, max_value=datetime.now().year, step=1)

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

# --- 3️⃣ 음성 전사 및 요약 ---
st.header("3️⃣ 음성 녹음 및 Whisper 전사")

if IS_LOCAL and st.button("🎙 5초간 녹음하기"):
    audio_np = record_audio(duration_sec=5)
    if audio_np is not None:
        wav_bytes = numpy_to_wav_bytes(audio_np)
        st.audio(wav_bytes, format="audio/wav")
        transcript, summary, script = transcribe_audio(model, wav_bytes)
        st.subheader("📝 전사 결과")
        st.write(transcript)
        st.subheader("🔍 요약")
        st.write(summary)
        st.subheader("🎬 감성 영상 스크립트")
        st.write(script)
        st.session_state["script"] = script  # ✅ 감성 스크립트 저장

uploaded_file = st.file_uploader("또는 오디오 파일(.wav/.mp3)을 업로드하세요", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    transcript, summary, script = transcribe_audio(model, uploaded_file)
    st.subheader("📝 전사 결과")
    st.write(transcript)
    st.subheader("🔍 요약")
    st.write(summary)
    st.subheader("🎬 감성 영상 스크립트")
    st.write(script)
    st.session_state["script"] = script  # ✅ 감성 스크립트 저장

# --- 4️⃣ 영상 생성 ---
st.header("4️⃣ 영상 생성")

prompt = st.session_state.get("video_prompt", None)
image_path = st.session_state.get("saved_image_path", None)
script = st.session_state.get("script", None)

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
