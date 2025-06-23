import streamlit as st
from PIL import Image
import io
import numpy as np
import wave
import tempfile
import os
import platform
from datetime import datetime
import openai

# --- 0. 기본 설정 및 함수 정의 ---

# 플랫폼 확인 (로컬/클라우드 구분)
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    try:
        import sounddevice as sd
    except Exception as e:
        st.error(f"Sounddevice 로드 실패. 마이크 사용이 불가할 수 있습니다: {e}")
        IS_LOCAL = False

# OpenAI API 키 설정
# 실제 배포 시에는 st.secrets 사용을 권장합니다.
# openai.api_key = st.secrets["OPENAI_API_KEY"] 
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
except Exception as e:
    st.error(f"API 키를 로드하는 중 오류가 발생했습니다: {e}")

# --- Helper Functions (GPT, Audio, Avatar) ---

# gpt_summarizer.py의 함수들
def summarize_text(text: str) -> str:
    prompt = f"Summarize the following interview transcript concisely in one or two sentences in Korean:\n\n{text}"
    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": "You are a helpful summarization expert."}, {"role": "user", "content": prompt}], max_tokens=300, temperature=0.5)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"요약 중 오류 발생: {e}"

def create_final_video_prompt(family_name: str, theme: str, transcript: str) -> str:
    prompt = f"""You are a creative and empathetic video director. Your task is to create a detailed, scene-by-scene storyboard prompt in English for an AI video generator.
    **Family Name:** {family_name}
    **Chosen Video Theme:** {theme}
    **Full Interview Transcript (in Korean):**\n---\n{transcript}\n---
    Based on all the information above, generate a rich, descriptive prompt that outlines a short video. Describe scenes, camera angles, emotions, and overall style to bring the family's story to life.
    Example: "Scene 1: A warm, sunlit kitchen. Close-up on the laughing face of '{family_name}' as they share a story. Style: nostalgic, warm, cinematic."
    """
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": "You are a professional video director creating prompts for an AI video generator."}, {"role": "user", "content": prompt}], max_tokens=500, temperature=0.7)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"프롬프트 생성 중 오류 발생: {e}"

# 오디오 처리 함수들
def record_audio(duration_sec=10, fs=16000):
    st.info(f"{duration_sec}초간 인터뷰 녹음을 시작합니다...")
    try:
        audio_data = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.success("녹음이 완료되었습니다!")
        return audio_data.flatten()
    except Exception as e:
        st.error(f"오디오 녹음 중 오류가 발생했습니다. 마이크 연결을 확인해주세요. 오류: {e}")
        return None

def numpy_to_wav_bytes(audio_np, fs=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs); wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_and_create_prompt(audio_input, family_name, theme):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_input.getvalue())
            tmp_path = tmp_file.name
        
        with open(tmp_path, "rb") as audio_file:
            result = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        transcript = result.text
        
        summary = summarize_text(transcript)
        final_prompt = create_final_video_prompt(family_name, theme, transcript)
        return transcript, summary, final_prompt
    except Exception as e:
        st.error(f"오디오 처리 중 오류 발생: {e}")
        return "", "", ""
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# 아바타 관련 함수 (실제 구현은 별도 파일에)
def extract_face(image_pil):
    # DUMMY: 실제 얼굴 추출 로직
    return image_pil.resize((256, 256))

def generate_avatar(prompt):
    # DUMMY: 실제 아바타 생성 로직
    return Image.new('RGB', (512, 512), color = 'red')

def create_video_from_text_and_image(full_prompt, image_path):
    st.info(f"영상 생성 중...\n\n🧾 프롬프트: {full_prompt}\n🖼️ 이미지 경로: {image_path}")
    st.success("✅ (예시) 영상 생성 완료! 실제 영상 생성 기능은 추후 연동됩니다.")


# --- 세션 상태 및 UI 초기화 ---
st.title("👨‍👩‍👧‍👦 가족 이야기 AI 영상 만들기")

if 'step' not in st.session_state:
    st.session_state.step = "select_theme"
# 다른 세션 변수들도 초기화
default_states = {
    "family_name": "", "selected_theme": "", "saved_image_path": None,
    "transcript": "", "summary": "", "final_prompt": None
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

os.makedirs("image_storage", exist_ok=True) # 이미지 저장 폴더 생성

# --- 1단계: 테마 선택 ---
if st.session_state.step == "select_theme":
    st.header("1단계: 영상 테마 정하기")
    family_name = st.text_input("가족의 호칭을 입력하세요", placeholder="예: 사랑하는 우리 가족")
    
    interview_questions = { "우리의 평범하지만 소중한 일상": [], "함께 떠났던 즐거운 여행의 추억": [], "특별한 날의 행복했던 순간들 (생일, 명절 등)": [], "아이들의 사랑스러운 성장 기록": [], "다시 봐도 웃음이 나는 우리 가족의 재미있는 순간": [], "서로에게 전하는 사랑과 감사의 메시지": [] }
    themes = list(interview_questions.keys())
    selected_theme = st.radio("어떤 테마의 영상을 만들고 싶으신가요?", themes, key="theme_radio")
    
    if st.button("다음 단계로: 얼굴 사진 입력 ▶️"):
        if not family_name:
            st.warning("가족의 호칭을 입력해주세요!")
        else:
            st.session_state.family_name = family_name
            st.session_state.selected_theme = selected_theme
            st.session_state.step = "capture_avatar"
            st.rerun()

# --- 2단계: 얼굴 사진(아바타) 입력 ---
elif st.session_state.step == "capture_avatar":
    st.header("2단계: 영상에 사용할 얼굴 사진 입력")
    st.info(f"**가족 호칭:** {st.session_state.family_name} / **테마:** {st.session_state.selected_theme}")

    tab1, tab2 = st.tabs(["📸 카메라 촬영", "🖼️ 이미지 업로드"])
    image_pil = None
    with tab1:
        if IS_LOCAL:
            image_file = st.camera_input("아바타용 사진을 찍어보세요")
            if image_file: image_pil = Image.open(image_file)
        else:
            st.warning("클라우드 환경에서는 카메라를 사용할 수 없습니다. 파일 업로드를 이용해주세요.")
    with tab2:
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
        if uploaded_file: image_pil = Image.open(uploaded_file)
    
    if image_pil:
        st.image(image_pil, caption="📷 원본 이미지", use_container_width=True)
        with st.spinner("얼굴을 인식하고 아바타를 생성하는 중..."):
            face_img = extract_face(image_pil)
            if face_img is None:
                st.error("😢 얼굴을 인식하지 못했습니다. 다른 이미지를 시도해주세요.")
            else:
                st.image(face_img, caption="✂️ 추출된 얼굴", width=256)
                # DUMMY 아바타 생성 로직, 실제로는 API 호출
                avatar_img = generate_avatar("dummy prompt") 
                st.image(avatar_img, caption="🖼️ 생성된 AI 아바타", use_container_width=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"image_storage/avatar_{timestamp}.jpg"
                avatar_img.save(save_path)
                st.session_state.saved_image_path = save_path
                st.success(f"✅ 아바타 이미지 저장 완료: {save_path}")

    if st.session_state.saved_image_path:
        if st.button("다음 단계로: 인터뷰 질문 확인 ▶️"):
            st.session_state.step = "show_questions"
            st.rerun()

# --- 3단계: 인터뷰 질문 확인 ---
elif st.session_state.step == "show_questions":
    st.header("3단계: 인터뷰 질문 확인")
    st.info(f"**가족 호칭:** {st.session_state.family_name} / **테마:** {st.session_state.selected_theme}")
    st.markdown("아래 질문들을 바탕으로 가족과 자유롭게 대화하며 인터뷰를 준비해주세요.")
    
    # 실제 질문 목록 표시
    # for i, q in enumerate(interview_questions[st.session_state.selected_theme]):
    #     st.markdown(f"**Q{i+1}.** {q}")

    if st.button("✅ 인터뷰 준비 완료! 녹음 시작하기 ▶️"):
        st.session_state.step = "record_interview"
        st.rerun()

# --- 4단계: 인터뷰 녹음 및 프롬프트 생성 ---
elif st.session_state.step == "record_interview":
    st.header("4단계: 인터뷰 녹음 및 AI 프롬프트 생성")
    st.info(f"**가족 호칭:** {st.session_state.family_name} / **테마:** {st.session_state.selected_theme}")
    
    record_duration = st.slider("녹음할 시간(초)을 선택하세요", 10, 180, 30)
    if IS_LOCAL and st.button(f"🎙️ {record_duration}초간 인터뷰 녹음 시작"):
        audio_np = record_audio(duration_sec=record_duration)
        if audio_np is not None:
            wav_bytes = numpy_to_wav_bytes(audio_np)
            st.audio(wav_bytes, format="audio/wav")
            with st.spinner("음성을 분석하고 영상 프롬프트를 생성하는 중..."):
                st.session_state.transcript, st.session_state.summary, st.session_state.final_prompt = \
                    transcribe_and_create_prompt(wav_bytes, st.session_state.family_name, st.session_state.selected_theme)

    uploaded_audio_file = st.file_uploader("또는 녹음된 인터뷰 파일(.wav/.mp3)을 업로드하세요", type=["wav", "mp3", "m4a"])
    if uploaded_audio_file:
        st.audio(uploaded_audio_file)
        with st.spinner("음성을 분석하고 영상 프롬프트를 생성하는 중..."):
            audio_bytes_io = io.BytesIO(uploaded_audio_file.getvalue())
            st.session_state.transcript, st.session_state.summary, st.session_state.final_prompt = \
                transcribe_and_create_prompt(audio_bytes_io, st.session_state.family_name, st.session_state.selected_theme)
    
    if st.session_state.final_prompt:
        st.success("✨ AI 프롬프트 생성이 완료되었습니다!")
        st.text_area("📝 인터뷰 전체 내용", st.session_state.transcript, height=150)
        st.text_area("🔍 한 줄 요약", st.session_state.summary, height=50)
        if st.button("다음 단계로: 최종 확인 및 영상 생성 ▶️"):
            st.session_state.step = "create_video"
            st.rerun()

# --- 5단계: 최종 확인 및 영상 생성 ---
elif st.session_state.step == "create_video":
    st.header("5단계: 최종 확인 및 영상 생성")
    
    image_path = st.session_state.saved_image_path
    final_prompt = st.session_state.final_prompt

    if image_path and final_prompt:
        st.info("아래 정보로 최종 영상을 생성합니다.")
        st.image(image_path, caption="🎨 최종 영상용 아바타 이미지")
        st.text_area("🎬 최종 영상 AI 프롬프트", final_prompt, height=200)
        if st.button("🎞️ 이 내용으로 영상 만들기", type="primary"):
            with st.spinner("영상을 생성 중입니다... (1~2분 소요될 수 있습니다)"):
                create_video_from_text_and_image(final_prompt, image_path)
    else:
        st.error("영상 생성에 필요한 정보가 부족합니다. 처음부터 다시 시작해주세요.")

# --- 처음으로 돌아가기 버튼 ---
if st.session_state.step != "select_theme":
    if st.button("처음으로 돌아가기"):
        # 모든 세션 상태를 초기값으로 리셋
        for key, value in default_states.items():
            st.session_state[key] = value
        st.session_state.step = "select_theme"
        st.rerun()