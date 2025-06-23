import os
import openai
from typing import List

# LangChain 및 Pydantic 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- API 키 설정 ---
# 이 파일을 단독으로 테스트할 경우를 대비해 포함합니다.
# app.py에서 import해서 사용할 때는 app.py의 설정이 적용됩니다.
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print(f"API 키 로드 중 오류 발생: {e}")


# --- ⭐️ 개별 Q&A 분석을 위한 함수 ---
def analyze_single_qa_pair(question: str, answer: str) -> bool:
    """
    하나의 질문과 답변 쌍을 분석하여, 답변이 충분히 구체적인지 (육하원칙 포함) 판단합니다.
    True 또는 False를 반환합니다.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    template = """
    You are a meticulous analyst. Your task is to determine if the provided 'Answer' is a good, detailed response to the 'Question'.
    A good answer is NOT abstract or generic. It is specific and contains core story elements (Who, What, Why).

    CRITICAL INSTRUCTION: Analyze ONLY the provided text. Do NOT infer or imagine information.

    - Question: "{question}"
    - Answer: "{answer}"

    Based on the answer's specificity and inclusion of story elements, respond with only "True" if the answer is sufficient, or "False" if it is too abstract or lacks detail.
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        result = chain.invoke({"question": question, "answer": answer})
        # LLM의 응답에서 'True' 또는 'False' 문자열을 실제 불리언 값으로 변환
        # .lower()를 통해 대소문자 구분 없이 처리
        return "true" in result['text'].lower()
    except Exception as e:
        print(f"개별 답변 분석 중 오류: {e}")
        return False # 오류 발생 시, 일단 불충분으로 간주


# --- 최종 프롬프트 생성 함수 ---
def create_final_video_prompt(family_name: str, theme: str, transcript: str) -> str:
    """모든 정보가 준비되었을 때, 최종 영상 프롬프트를 생성합니다."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    
    template = """
    You are a creative video director. Your task is to create a detailed, scene-by-scene storyboard prompt in English for an AI video generator.

    **Family Name:** {family_name}
    **Chosen Video Theme:** {theme}
    **Full Interview Transcript (in Korean):**
    ---
    {transcript}
    ---
    Based on all information, generate a rich, descriptive prompt that outlines a short video. Describe scenes, camera angles, character emotions, and overall style to bring the family's story to life.
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.invoke({"family_name": family_name, "theme": theme, "transcript": transcript})
        return result['text'].strip()
    except Exception as e:
        return f"최종 프롬프트 생성 중 오류: {e}"

# --- analyze_transcript_for_completeness 함수는 더 이상 사용되지 않으므로 삭제 또는 주석 처리 ---
# class AnalysisResult(BaseModel): ...
# def analyze_transcript_for_completeness(transcript: str) -> AnalysisResult: ...
