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

# --------------------------------------------------------------------------
# --- 1. gpt_summarizer.py에서 함수 import ---
# --------------------------------------------------------------------------
try:
    from summarizer.gpt_summarizer import analyze_transcript_for_completeness, create_final_video_prompt
except ImportError:
    st.error("오류: gpt_summarizer.py 파일을 찾을 수 없습니다. app.py와 같은 폴더에 있는지 확인해주세요.")
    st.stop()

# --------------------------------------------------------------------------
# --- 2. 초기 설정 및 모든 헬퍼 함수 정의 ---
# --------------------------------------------------------------------------

# 플랫폼 확인 (로컬에서만 마이크 사용)
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    try:
        import sounddevice as sd
    except Exception:
        IS_LOCAL = False

# 이미지 저장 폴더 생성
os.makedirs("image_storage", exist_ok=True)

# --- 모든 헬퍼 함수 (Audio, Dummy, etc.) ---

def record_audio(duration_sec=10, fs=16000):
    """지정된 시간 동안 마이크에서 오디오를 녹음합니다."""
    st.info(f"{duration_sec}초간 인터뷰 녹음을 시작합니다...")
    try:
        audio_data = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.success("녹음이 완료되었습니다!")
        return audio_data.flatten()
    except Exception as e:
        st.error(f"오디오 녹음 중 오류가 발생했습니다: {e}"); return None

def numpy_to_wav_bytes(audio_np, fs=16000):
    """Numpy 오디오 배열을 WAV 형식의 BytesIO 객체로 변환합니다."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs); wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio_from_bytes(audio_bytes_io):
    """BytesIO 오디오 객체를 Whisper API로 전사합니다."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes_io.getvalue())
            tmp_path = tmp_file.name
        with open(tmp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
        return str(transcript)
    except Exception as e:
        st.error(f"음성 전사 중 오류 발생: {e}"); return ""
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)


# --- DUMMY Functions (실제 서비스 연동이 필요한 부분) ---
def extract_face(image_pil):
    """DUMMY: 이미지에서 얼굴 부분을 추출하는 함수입니다."""
    return image_pil.resize((256, 256))

def generate_avatar(face_image):
    """DUMMY: AI 아바타를 생성하는 함수입니다."""
    return Image.new('RGB', (512, 512), color = 'purple')

def create_video_from_text_and_image(prompt, image_path):
    """DUMMY: 최종 프롬프트와 이미지로 영상을 생성하는 함수입니다."""
    st.success("✅ (예시) 영상 생성 완료!");


# --- 예시 질문 목록 ---
interview_questions = {
    "우리의 평범하지만 소중한 일상": ["가족과 함께하는 아침 식사 시간은 어떤 모습인가요?", "최근에 함께 산책하며 나눈 소소한 대화가 있다면 알려주세요.", "우리 가족만의 자기 전 습관이나 주말을 보내는 특별한 방법이 있나요?"],
    "함께 떠났던 즐거운 여행의 추억": ["지금까지 가장 기억에 남는 가족 여행은 어디였나요? 왜 그곳이 특별했나요?", "여행지에서 겪었던 가장 재미있는 에피소드나 예상치 못한 사건이 있었나요?", "사진을 다시 펼쳐보게 되는, 가장 마음에 드는 여행 사진은 무엇인가요?"],
    "특별한 날의 행복했던 순간들": ["가장 기억에 남는 생일이나 기념일은 언제였나요?", "우리 가족만이 가진 특별한 명절 전통이 있다면 무엇인가요?", "서로에게 줬던 선물 중 가장 감동적이거나 재미있었던 것은 무엇이었나요?"],
    "아이들의 사랑스러운 성장 기록": ["자녀가 태어났을 때, 혹은 처음으로 '엄마/아빠'라고 불렀을 때의 기분이 어땠나요?", "학창 시절, 가장 자랑스러웠던 순간이나 큰 도전을 했던 기억이 있나요?", "부모님께서는 자녀가 성장하는 모습을 보며 언제가 가장 대견하고 뿌듯했나요?"],
    "다시 봐도 웃음이 나는 우리 가족의 재미있는 순간": ["우리 가족 구성원들만 아는 서로의 재미있는 습관이나 버릇이 있나요?", "생각만 해도 웃음이 나는 우리 가족의 '흑역사'나 재미있는 실수담이 있다면?", "가장 성공적이었던 (또는 완전히 실패했던) 가족 장난은 무엇이었나요?"],
    "서로에게 전하는 사랑과 감사의 메시지": ["가족이어서 가장 힘이 되었던 순간은 언제였나요?", "서로에게 평소에 쑥스러워서 하지 못했던 고마운 마음을 표현해주세요.", "앞으로 우리 가족이 함께 이루고 싶은 꿈이나 소망이 있다면 무엇인가요?"]
}

# --------------------------------------------------------------------------
# --- 3. Streamlit UI 및 상태 관리 ---
# --------------------------------------------------------------------------

st.title("👨‍👩‍👧‍👦 가족 이야기 AI 영상 만들기")

# --- 세션 상태 초기화 ---
default_states = {
    "step": "select_theme", "family_name": "", "selected_theme": list(interview_questions.keys())[0],
    "saved_image_path": None, "final_prompt": None,
    "qa_list": [], "current_question": "", "transcript": ""
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 단계별 UI 렌더링 ---

if st.session_state.step == "select_theme":
    st.header("1단계: 영상 테마 정하기")
    st.session_state.family_name = st.text_input("가족의 호칭을 입력하세요", value=st.session_state.family_name, placeholder="예: 사랑하는 우리 가족")
    themes = list(interview_questions.keys())
    st.session_state.selected_theme = st.radio("어떤 테마의 영상을 만들고 싶으신가요?", themes, index=themes.index(st.session_state.selected_theme))
    
    if st.button("다음 단계로: 얼굴 사진 입력 ▶️", type="primary"):
        if not st.session_state.family_name:
            st.warning("가족의 호칭을 입력해주세요!")
        else:
            st.session_state.step = "capture_avatar"; st.rerun()

elif st.session_state.step == "capture_avatar":
    st.header("2단계: 영상에 사용할 얼굴 사진 입력")
    image_pil = None
    if IS_LOCAL:
        tab1, tab2 = st.tabs(["📸 카메라 촬영", "🖼️ 이미지 업로드"])
        with tab1: image_file = st.camera_input("아바타용 사진을 찍어보세요")
        with tab2: uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    else:
        st.warning("클라우드 환경에서는 카메라를 사용할 수 없습니다. 파일 업로드만 가능합니다.")
        uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
        image_file = None
    
    if image_file if 'image_file' in locals() and image_file is not None else None: image_pil = Image.open(image_file)
    if uploaded_file: image_pil = Image.open(uploaded_file)

    if image_pil:
        with st.spinner("얼굴을 인식하고 아바타를 생성하는 중..."):
            face_img = extract_face(image_pil)
            avatar_img = generate_avatar(face_img)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"image_storage/avatar_{timestamp}.jpg"
            avatar_img.save(save_path)
            st.session_state.saved_image_path = save_path
        st.success(f"아바타 이미지 저장 완료!"); st.image(avatar_img, caption="생성된 AI 아바타")

    if st.session_state.saved_image_path:
        if st.button("다음 단계로: 인터뷰 준비 ▶️", type="primary"):
            st.session_state.step = "show_questions"; st.rerun()

elif st.session_state.step == "show_questions":
    st.header("3단계: 인터뷰 준비")
    st.info(f"**선택 테마:** {st.session_state.selected_theme}")
    st.markdown("아래 질문들을 보며 어떤 이야기를 할지 자유롭게 구상해보세요.")
    questions = interview_questions.get(st.session_state.selected_theme, [])
    for i, q in enumerate(questions): st.markdown(f"**Q{i+1}.** {q}")
    if st.button("✅ 준비 완료! 녹음 시작하기 ▶️", type="primary"):
        st.session_state.step = "record_interview"; st.rerun()

elif st.session_state.step == "record_interview":
    st.header("4단계: 대화형 인터뷰 진행")

    with st.expander("📖 선택한 테마의 예시 질문 다시보기", expanded=True):
        questions = interview_questions.get(st.session_state.selected_theme, [])
        for q in questions: st.markdown(f"- {q}")
    st.markdown("---")

    if st.session_state.qa_list:
        st.subheader("✅ 완성된 질문/답변 목록")
        for i, qa in enumerate(st.session_state.qa_list):
            with st.container(border=True):
                st.markdown(f"**Q{i+1}.** {qa['question']}")
                st.markdown(f"**A{i+1}.** {qa['answer']}")
        st.markdown("---")

    st.subheader("➕ 새로운 질문 & 답변 추가하기")

    if not st.session_state.current_question:
        st.markdown("**1. 먼저 질문을 녹음하세요.**")
        if IS_LOCAL and st.button("🎙️ 질문 녹음하기 (5초)"):
            with st.spinner("질문을 녹음하고 변환 중..."):
                audio_np = record_audio(duration_sec=5)
                if audio_np is not None:
                    wav_bytes = numpy_to_wav_bytes(audio_np)
                    st.session_state.current_question = transcribe_audio_from_bytes(wav_bytes)
                    st.rerun()
    else:
        st.success(f"**녹음된 질문:** {st.session_state.current_question}")
        st.markdown("**2. 이제 위 질문에 대한 답변을 녹음하세요.**")

        with st.container(border=True):
            st.markdown("💡 **답변 Tip: 풍부한 이야기를 위한 육하원칙!**")
            st.caption("답변하실 때 아래 내용을 포함하면 더 생생하고 감동적인 이야기가 돼요.")
            st.markdown("- **누가 (Who):** 이야기의 주인공은 누구인가요?\n- **언제 (When):** 그 일은 언제 있었나요?\n- **어디서 (Where):** 어떤 장소였나요?\n- **무엇을 (What):** 어떤 특별한 사건이 있었나요?\n- **왜 (Why):** 그 순간이 왜 중요하고 특별했나요?\n- **어떻게 (How):** 당시의 분위기나 감정은 어땠나요?")
        
        record_duration = st.slider("답변 녹음 시간(초)", 10, 180, 30, key="answer_duration")
        if IS_LOCAL and st.button(f"🎤 답변 녹음하기 ({record_duration}초)"):
            with st.spinner("답변을 녹음하고 변환 중..."):
                audio_np = record_audio(duration_sec=record_duration)
                if audio_np is not None:
                    wav_bytes = numpy_to_wav_bytes(audio_np)
                    answer = transcribe_audio_from_bytes(wav_bytes)
                    if answer:
                        st.session_state.qa_list.append({"question": st.session_state.current_question, "answer": answer})
                        st.session_state.current_question = ""
                        st.rerun()
    st.markdown("---")

    if st.session_state.qa_list:
        if st.button("✅ 인터뷰 완료 및 분석 시작", type="primary"):
            with st.spinner("전체 인터뷰 내용을 종합하여 분석 중입니다..."):
                full_transcript = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in st.session_state.qa_list])
                st.session_state.transcript = full_transcript
                analysis_result = analyze_transcript_for_completeness(full_transcript)
                if analysis_result.is_complete:
                    final_prompt = create_final_video_prompt(st.session_state.family_name, st.session_state.selected_theme, full_transcript)
                    st.session_state.final_prompt = final_prompt
                else:
                    st.warning("⚠️ 이야기의 핵심 요소(누가, 무엇을, 왜)가 부족합니다.\n\n질문과 답변을 더 추가하여 이야기를 구체화해주세요.")
            if st.session_state.final_prompt:
                st.success("충분한 내용이 확인되었습니다! 최종 프롬프트를 생성합니다.")
                st.rerun()
    
    if st.session_state.final_prompt:
        st.success("✨ AI 프롬프트 생성이 완료되었습니다!")
        st.text_area("생성된 최종 AI 프롬프트", st.session_state.final_prompt, height=200)
        if st.button("다음 단계로: 최종 확인 ▶️"):
            st.session_state.step = "create_video"; st.rerun()

elif st.session_state.step == "create_video":
    st.header("5단계: 최종 확인 및 영상 생성")
    if st.session_state.saved_image_path and st.session_state.final_prompt:
        st.info("아래 정보로 최종 영상을 생성합니다.")
        st.image(st.session_state.saved_image_path, caption="🎨 최종 영상용 아바타 이미지")
        st.text_area("🎬 최종 영상 AI 프롬프트", st.session_state.final_prompt, height=200)
        if st.button("🎞️ 이 내용으로 영상 만들기", type="primary"):
            with st.spinner("영상을 생성 중입니다..."):
                create_video_from_text_and_image(st.session_state.final_prompt, st.session_state.saved_image_path)
    else:
        st.error("영상 생성에 필요한 정보가 부족합니다. 처음부터 다시 시작해주세요.")

if st.session_state.step != "select_theme":
    if st.button("처음으로 돌아가기"):
        for key in default_states.keys():
            st.session_state[key] = default_states[key]
        st.rerun()

