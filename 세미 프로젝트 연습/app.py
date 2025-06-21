# app.py
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

from camera.face_capture import extract_face
from summarizer.gpt_summarizer import summarize_text, generate_video_script

# --- 플랫폼 확인 (로컬/클라우드 구분) ---
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    import sounddevice as sd

# Whisper 모델 캐시 로드
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# 오디오 녹음 함수 (로컬 전용)
def record_audio(duration_sec=5, fs=16000, device=None):
    if not IS_LOCAL:
        st.error("⚠️ 이 기능은 로컬 환경에서만 작동합니다. Streamlit Cloud에서는 음성 파일을 업로드해 주세요.")
        return None
    st.info(f"{duration_sec}초간 녹음 시작...")
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    st.success("녹음 완료!")
    return audio.flatten()

# NumPy 배열 -> WAV 바이트 스트림 변환
def numpy_to_wav_bytes(audio_np, fs=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

# Whisper 전사 및 요약/스크립트 생성
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

# Streamlit 설정 및 제목 표시
st.set_page_config(page_title="AI 아바타 + 음성 녹음 & 전사", layout="centered")
st.title("📸 AI 아바타 생성 + 🎤 음성 녹음 & Whisper 전사")

# --- 1️⃣ 사진 촬영 및 얼굴 추출 ---
st.header("1️⃣ 사진 촬영 및 얼굴 추출")
image_file = st.camera_input("아바타용 사진을 찍어보세요")

if image_file:
    image_pil = Image.open(image_file)
    st.image(image_pil, caption="📷 촬영된 원본 이미지", use_container_width=True)

    face_img = extract_face(image_pil)
    if face_img is None:
        st.error("😢 얼굴을 인식하지 못했습니다. 다시 시도해주세요.")
    else:
        st.image(face_img, caption="✂️ 얼굴 영역 추출", width=256)
        st.write("🎨 AI 아바타 생성 (임시 버전)")
        avatar_img = face_img
        st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True)

        save_dir = "C:/Users/user/Desktop/세미 프로젝트 연습/image"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"face_{timestamp}.jpg")
        avatar_img.save(save_path)
        st.success(f"얼굴 이미지가 저장되었습니다:\n{save_path}")

# --- 2️⃣ 음성 녹음 및 Whisper 전사 ---
st.header("2️⃣ 음성 녹음 및 Whisper 전사")

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

# 파일 업로드 (클라우드 대응)
uploaded_file = st.file_uploader("또는 오디오 파일(.wav/.mp3)을 업로드하세요", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    transcript, summary, script = transcribe_audio(model, uploaded_file)
    st.subheader("📝 전사 결과")
    st.write(transcript)
    st.subheader("🔍 요약")
    st.write(summary)
    st.subheader("🎬 감성 영상 스크립트")
    st.write(script)
