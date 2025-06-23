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
from typing import List

# LangChain 및 Pydantic 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# --- 0. 초기 설정: 플랫폼 확인, API 키 로드, 폴더 생성 ---
# --------------------------------------------------------------------------

# 플랫폼 확인 (로컬 환경에서만 마이크 녹음 기능 활성화)
IS_LOCAL = platform.system() != "Linux"
if IS_LOCAL:
    try:
        import sounddevice as sd
    except Exception as e:
        st.error(f"Sounddevice 로드 실패. 마이크 사용이 불가할 수 있습니다: {e}")
        IS_LOCAL = False

# OpenAI API 키 설정
# 로컬 테스트 시: 컴퓨터 환경 변수에 'OPENAI_API_KEY' 설정 필요
# Streamlit 클라우드 배포 시: st.secrets["OPENAI_API_KEY"] 사용 권장
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 키를 설정해주세요.")
except Exception as e:
    st.error(f"API 키를 로드하는 중 오류가 발생했습니다: {e}")

# 이미지 저장 폴더 생성
os.makedirs("image_storage", exist_ok=True)


# --------------------------------------------------------------------------
# --- 1. 핵심 로직 함수 정의 (GPT, LangChain, 오디오, 아바타 등) ---
# --------------------------------------------------------------------------

# --- GPT & LangChain Functions ---

class AnalysisResult(BaseModel):
    """육하원칙 분석 결과를 담는 데이터 구조"""
    is_complete: bool = Field(description="인터뷰 내용에 육하원칙의 핵심 요소(누가, 무엇을, 왜)가 모두 포함되어 있는지 여부")
    missing_elements: List[str] = Field(description="누락된 육하원칙 요소 목록 (한국어). 예: ['언제', '왜']")
    summary: str = Field(description="현재까지 파악된 인터뷰 내용의 간략한 요약")

def analyze_transcript_for_completeness(transcript: str) -> AnalysisResult:
    """인터뷰 내용을 '창의적 추론 없이' 엄격하게 분석하여 완전성 여부를 진단합니다."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=AnalysisResult)
    template = """
    You are a meticulous analyst. Your task is to strictly analyze an interview transcript based on the 5W1H principle (육하원칙) and determine if it contains enough information.
    A story is considered "complete" ONLY if it clearly contains the core elements: Who, What, and Why.
    CRITICAL INSTRUCTION: Do NOT infer or create any information that is not explicitly present in the transcript.
    Transcript:
    ---
    {transcript}
    ---
    Based *only* on the transcript provided, provide your analysis in the following JSON format.
    {format_instructions}
    """
    prompt = PromptTemplate(template=template, input_variables=["transcript"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        output = chain.invoke({"transcript": transcript})
        return parser.parse(output['text'])
    except Exception:
        return AnalysisResult(is_complete=True, missing_elements=[], summary="분석 중 오류 발생")

def generate_follow_up_question(summary: str, missing_elements: List[str]) -> str:
    """분석 결과를 바탕으로 사용자에게 할 자연스러운 추가 질문을 생성합니다."""
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    missing_elements_str = ', '.join([f"'{el}'" for el in missing_elements])
    template = """You are a friendly interviewer. A user's story summary is "{summary}". It's missing these elements: {missing_elements_str}.
    Based on this, generate one friendly follow-up question in Korean to get the missing details."""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.invoke({"summary": summary, "missing_elements_str": missing_elements_str})
        return result['text'].strip()
    except Exception as e:
        return f"추가 질문 생성 중 오류: {e}"

def create_final_video_prompt(family_name: str, theme: str, transcript: str) -> str:
    """모든 정보를 종합하여 최종 영상 프롬프트를 생성합니다."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    template = """You are a creative video director. Create a detailed, scene-by-scene storyboard prompt in English.
    Family Name: {family_name}
    Theme: {theme}
    Transcript: {transcript}
    Based on all info, generate a rich prompt describing scenes, camera angles, emotions, and style."""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.invoke({"family_name": family_name, "theme": theme, "transcript": transcript})
        return result['text'].strip()
    except Exception as e:
        return f"최종 프롬프트 생성 중 오류: {e}"


# --- Audio Processing Functions ---
def record_audio(duration_sec=10, fs=16000):
    st.info(f"{duration_sec}초간 인터뷰 녹음을 시작합니다...")
    try:
        audio_data = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        st.success("녹음이 완료되었습니다!")
        return audio_data.flatten()
    except Exception as e:
        st.error(f"오디오 녹음 중 오류가 발생했습니다: {e}"); return None

def numpy_to_wav_bytes(audio_np, fs=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio(audio_input):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_input.getvalue())
            tmp_path = tmp_file.name
        with open(tmp_path, "rb") as audio_file:
            result = openai.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
        return str(result)
    except Exception as e:
        st.error(f"음성 전사 중 오류 발생: {e}"); return ""
    finally:
        if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)


# --- DUMMY Functions (실제 구현 필요) ---
def extract_face(image_pil):
    st.info("DUMMY: 얼굴 추출 로직 실행 중...")
    return image_pil.resize((256, 256))

def generate_avatar(face_image):
    st.info("DUMMY: AI 아바타 생성 로직 실행 중...")
    return Image.new('RGB', (512, 512), color = 'blue')

def create_video_from_text_and_image(prompt, image_path):
    st.info(f"DUMMY: 영상 생성 중...\n- 프롬프트: {prompt[:50]}...\n- 이미지 경로: {image_path}")
    st.success("✅ (예시) 영상 생성 완료!")
    # 여기에 실제 영상 생성 API(D-ID, RunwayML 등) 호출 로직 구현


# --------------------------------------------------------------------------
# --- 2. Streamlit UI 및 상태 관리 ---
# --------------------------------------------------------------------------

st.title("👨‍👩‍👧‍👦 가족 이야기 AI 영상 만들기")

# --- 세션 상태 초기화 ---
default_states = {
    "step": "select_theme", "family_name": "", "selected_theme": "",
    "saved_image_path": None, "transcript": "", "final_prompt": None,
    "follow_up_question": None
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 단계별 UI 렌더링 ---

# 1단계: 테마 선택
if st.session_state.step == "select_theme":
    st.header("1단계: 영상 테마 정하기")
    st.session_state.family_name = st.text_input("가족의 호칭을 입력하세요", value=st.session_state.family_name, placeholder="예: 사랑하는 우리 가족")
    themes = ["우리의 평범하지만 소중한 일상", "함께 떠났던 즐거운 여행의 추억", "특별한 날의 행복했던 순간들", "아이들의 사랑스러운 성장 기록", "다시 봐도 웃음이 나는 우리 가족의 재미있는 순간", "서로에게 전하는 사랑과 감사의 메시지"]
    st.session_state.selected_theme = st.radio("어떤 테마의 영상을 만들고 싶으신가요?", themes, key="theme_radio")
    
    if st.button("다음 단계로: 얼굴 사진 입력 ▶️", type="primary"):
        if not st.session_state.family_name:
            st.warning("가족의 호칭을 입력해주세요!")
        else:
            st.session_state.step = "capture_avatar"; st.rerun()

# 2단계: 아바타 생성
elif st.session_state.step == "capture_avatar":
    st.header("2단계: 영상에 사용할 얼굴 사진 입력")
    image_pil = None
    if IS_LOCAL:
        tab1, tab2 = st.tabs(["📸 카메라 촬영", "🖼️ 이미지 업로드"])
        with tab1:
            image_file = st.camera_input("아바타용 사진을 찍어보세요");
            if image_file: image_pil = Image.open(image_file)
        with tab2:
            uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"]);
            if uploaded_file: image_pil = Image.open(uploaded_file)
    else:
        st.warning("클라우드 환경에서는 카메라를 사용할 수 없습니다.")
        uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"]);
        if uploaded_file: image_pil = Image.open(uploaded_file)

    if image_pil:
        with st.spinner("얼굴을 인식하고 아바타를 생성하는 중..."):
            face_img = extract_face(image_pil)
            avatar_img = generate_avatar(face_img)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"image_storage/avatar_{timestamp}.jpg"
            avatar_img.save(save_path)
            st.session_state.saved_image_path = save_path
        st.success(f"아바타 이미지 저장 완료!"); st.image(avatar_img)

    if st.session_state.saved_image_path:
        if st.button("다음 단계로: 인터뷰 준비 ▶️", type="primary"):
            st.session_state.step = "show_questions"; st.rerun()

# 3단계: 인터뷰 질문 확인
elif st.session_state.step == "show_questions":
    st.header("3단계: 인터뷰 준비")
    st.info(f"**테마: {st.session_state.selected_theme}**")
    st.markdown("아래와 같은 질문들을 바탕으로 가족과 자유롭게 대화하며 인터뷰를 준비해주세요.\n\n(예시 질문: `가장 기억에 남는 순간은 언제였나요?`, `그때 기분이 어땠나요?`)")
    if st.button("✅ 준비 완료! 녹음 시작하기 ▶️", type="primary"):
        st.session_state.step = "record_interview"; st.rerun()

# 4단계: 인터뷰 녹음 및 대화형 분석
elif st.session_state.step == "record_interview":
    st.header("4단계: 인터뷰 녹음 및 AI 분석")
    
    audio_bytes_io = None
    if IS_LOCAL:
        record_duration = st.slider("녹음할 시간(초)", 10, 180, 30)
        if st.button(f"🎙️ {record_duration}초간 인터뷰 녹음 시작"):
            audio_np = record_audio(duration_sec=record_duration)
            if audio_np is not None: audio_bytes_io = numpy_to_wav_bytes(audio_np)
    
    uploaded_audio_file = st.file_uploader("또는 녹음된 인터뷰 파일 업로드", type=["wav", "mp3", "m4a"])
    if uploaded_audio_file: audio_bytes_io = io.BytesIO(uploaded_audio_file.getvalue())

    if audio_bytes_io:
        st.audio(audio_bytes_io)
        with st.spinner("음성을 텍스트로 변환 중..."):
            transcript = transcribe_audio(audio_bytes_io)
            st.session_state.transcript = st.session_state.transcript + "\n" + transcript if st.session_state.transcript else transcript
            st.info("음성 변환이 완료되었습니다. 내용을 분석합니다.")

    if st.session_state.transcript and not st.session_state.final_prompt:
        st.text_area("현재까지의 인터뷰 내용", st.session_state.transcript, height=150)
        with st.spinner("답변 내용을 분석하여 다음 단계를 확인합니다..."):
            analysis_result = analyze_transcript_for_completeness(st.session_state.transcript)
            if analysis_result.is_complete:
                st.session_state.final_prompt = create_final_video_prompt(st.session_state.family_name, st.session_state.selected_theme, st.session_state.transcript)
            else:
                st.session_state.follow_up_question = generate_follow_up_question(analysis_result.summary, analysis_result.missing_elements)
        st.rerun()

    if st.session_state.follow_up_question:
        st.warning(st.session_state.follow_up_question)
        if st.button("🎤 추가 답변 녹음하기"):
            st.session_state.follow_up_question = None; st.rerun()

    if st.session_state.final_prompt:
        st.success("✨ 모든 정보가 준비되었습니다!")
        if st.button("다음 단계로: 최종 확인 ▶️", type="primary"):
            st.session_state.step = "create_video"; st.rerun()

# 5단계: 최종 확인 및 영상 생성
elif st.session_state.step == "create_video":
    st.header("5단계: 최종 확인 및 영상 생성")
    if st.session_state.saved_image_path and st.session_state.final_prompt:
        st.info("아래 정보로 최종 영상을 생성합니다.")
        st.image(st.session_state.saved_image_path, caption="🎨 최종 영상용 아바타 이미지")
        st.text_area("🎬 최종 영상 AI 프롬프트", st.session_state.final_prompt, height=200)
        if st.button("🎞️ 이 내용으로 영상 만들기", type="primary"):
            with st.spinner("영상을 생성 중입니다... (1~2분 소요될 수 있습니다)"):
                create_video_from_text_and_image(st.session_state.final_prompt, st.session_state.saved_image_path)
    else:
        st.error("영상 생성에 필요한 정보가 부족합니다. 처음부터 다시 시작해주세요.")

# --- '처음으로' 버튼 ---
if st.session_state.step != "select_theme":
    if st.button("처음으로 돌아가기"):
        for key in default_states.keys():
            st.session_state[key] = default_states[key]
        st.rerun()