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

from avatar_create.avatar_generator import generate_avatar_image, download_image_from_url, generate_avatar
from camera.face_capture import extract_face
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
st.set_page_config(page_title="공감 필름", layout="centered")
st.title("📸 공감 필름")

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

# --- 👨‍👩‍👧‍👦 가족 이름 입력 및 테마 선택 ---
st.title("가족 이야기 영상 프롬프트 생성기 👨‍👩‍👧‍👦")

# 1. 가족 이름 입력
family_name = st.text_input("가족의 호칭을 입력하세요 (예: 사랑하는 우리 가족, 행복한 김씨네)")

# 2. 영상 테마 6가지 정의
themes = [
    "우리의 평범하지만 소중한 일상",
    "함께 떠났던 즐거운 여행의 추억",
    "특별한 날의 행복했던 순간들 (생일, 명절 등)",
    "아이들의 사랑스러운 성장 기록",
    "다시 봐도 웃음이 나는 우리 가족의 재미있는 순간",
    "서로에게 전하는 사랑과 감사의 메시지"
]

# 3. 라디오 버튼으로 테마 선택
selected_theme = st.radio(
    "어떤 테마의 영상을 만들고 싶으신가요?",
    themes,
    # index=0 # 기본 선택값을 첫 번째 옵션으로 설정
)

# 4. 프롬프트 생성 버튼
if st.button("프롬프트 생성하기"):
    # 가족 호칭을 입력했는지 확인
    if not family_name:
        st.warning("가족의 호칭을 입력해주세요!")
    else:
        # 선택된 테마에 따라 맞춤형 프롬프트 생성
        if selected_theme == themes[0]: # 우리의 평범하지만 소중한 일상
            prompt = f"'{family_name}'의 소소한 행복이 담긴 일상을 따뜻하고 감성적인 영상으로 만들어줘. 아침 식사, 함께하는 산책, 저녁의 대화 같은 장면을 중심으로."

        elif selected_theme == themes[1]: # 함께 떠났던 즐거운 여행의 추억
            prompt = f"'{family_name}'이 함께 떠났던 여행의 순간들을 모아 경쾌하고 신나는 영상으로 만들어줘. 아름다운 풍경과 가족들의 웃음소리가 가득하게."

        elif selected_theme == themes[2]: # 특별한 날의 행복했던 순간들
            prompt = f"'{family_name}'의 생일 파티, 기념일, 명절 등 특별했던 날의 기억들을 모아 행복이 넘치는 축제 분위기의 영상으로 제작해줘."

        elif selected_theme == themes[3]: # 아이들의 사랑스러운 성장 기록
            prompt = f"'{family_name}' 아이들의 사랑스러운 성장 과정을 담은 영상. 첫 걸음마부터 입학식까지, 감동적인 순간들을 시간 순서대로 보여줘."

        elif selected_theme == themes[4]: # 다시 봐도 웃음이 나는 우리 가족의 재미있는 순간
            prompt = f"'{family_name}'의 배꼽 빠지는 재미있는 실수나 장난들을 모아서 유쾌하고 코믹한 시트콤 같은 영상으로 만들어줘. 웃음 효과음도 넣어줘."

        elif selected_theme == themes[5]: # 서로에게 전하는 사랑과 감사의 메시지
            prompt = f"'{family_name}' 구성원들이 서로에게 전하는 진심 어린 사랑과 감사의 마음을 담은 뭉클한 영상. 잔잔한 배경 음악과 함께 따뜻한 메시지를 자막으로 넣어줘."

        # 생성된 프롬프트 보여주기
        st.info(f"✅ 생성된 영상 프롬프트입니다:")
        st.write(f"**{prompt}**")

        # 세션 상태에 프롬프트 저장 (다른 페이지나 기능에서 사용하기 위함)
        st.session_state["video_prompt"] = prompt

# --- 2️⃣ 사진 입력: 카메라 촬영 또는 이미지 업로드 ---
st.header("2️⃣ 사진 입력: 카메라 촬영 또는 이미지 업로드")

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

if image_pil:
    gender = st.radio("이 사진 속 인물의 성별은?", ["남자", "여자"], horizontal=True)
    face_img = None
    try:
        face_img = extract_face(image_pil)
    except Exception as e:
        st.error(f"얼굴 추출 중 오류 발생: {e}")
        face_img = None

    if face_img is None:
        st.error("😢 얼굴을 인식하지 못했습니다. 다른 이미지를 시도해주세요.")
    else:
        st.image(face_img, caption="✂️ 추출된 얼굴", width=256)

        if gender == "남자":
            avatar_generation_prompt = (
                "Create a photorealistic studio portrait of a Korean man based on the original image. "
                "Preserve all facial features and structure, ensuring the identity and ethnicity are clearly Korean. "
                "Enhance his appearance subtly: smooth skin texture, defined jawline, sharp expressive eyes, and soft, natural lighting. "
                "Maintain a masculine look with a confident but approachable expression. "
                "Do not exaggerate, stylize, or westernize. High-resolution, true-to-life realism."
            )

        else:
            avatar_generation_prompt = (
                "Create a photorealistic studio portrait of a Korean woman based on the original image. "
                "Keep her facial structure, identity, and Korean ethnicity clearly intact. "
                "Gently enhance her features: smooth glowing skin, brightened eyes, soft expression, and flattering lighting. "
                "Maintain a feminine and natural appearance, avoiding stylization or exaggeration. "
                "The result should look like a refined studio photo of a real Korean woman. High resolution, true-to-life realism."
            )


        avatar_img = None
        try:
            avatar_img = generate_avatar(avatar_generation_prompt)
        except Exception as e:
            st.error(f"아바타 이미지 생성 오류: {e}")

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