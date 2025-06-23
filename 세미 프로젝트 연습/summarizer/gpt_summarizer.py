# gpt_summarizer.py 파일

import os
import openai
from typing import List

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- API 키 설정 ---
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception as e:
    print(f"API 키 로드 중 오류 발생: {e}")

# --- 분석가를 위한 출력 구조 정의 ---
class AnalysisResult(BaseModel):
    is_complete: bool = Field(description="인터뷰 내용에 육하원칙의 핵심 요소(누가, 무엇을, 왜)가 모두 포함되어 있는지 여부")
    summary: str = Field(description="현재까지 파악된 인터뷰 내용의 간략한 요약")

# --- 분석가 체인 함수 ---
def analyze_transcript_for_completeness(transcript: str) -> AnalysisResult:
    """인터뷰 내용을 엄격하게 분석하여 완전성 여부를 진단합니다."""
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=AnalysisResult)
    template = """
    You are a meticulous analyst. Your task is to strictly analyze an interview transcript.
    A story is considered "complete" ONLY if it clearly contains the core elements: Who, What, and Why.
    CRITICAL INSTRUCTION: Do NOT infer or create any information. Your job is ONLY to analyze the provided text.
    Transcript: --- {transcript} ---
    Based *only* on the transcript, provide your analysis in the following JSON format.
    {format_instructions}
    """
    prompt = PromptTemplate(template=template, input_variables=["transcript"], partial_variables={"format_instructions": parser.get_format_instructions()})
    chain = LLMChain(llm=llm, prompt=prompt)
    try:
        output = chain.invoke({"transcript": transcript})
        return parser.parse(output['text'])
    except Exception as e:
        print(f"분석 체인 실행 중 오류: {e}")
        return AnalysisResult(is_complete=True, summary="분석 중 오류 발생")

# --- 최종 프롬프트 생성 함수 ---
def create_final_video_prompt(family_name: str, theme: str, transcript: str) -> str:
    """모든 정보가 준비되었을 때, 최종 영상 프롬프트를 생성합니다."""
    # (이 함수는 변경 없이 그대로 사용합니다)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
    template = "..." # (이전과 동일)
    # ...
    try:
        # ...
        return "..." # (이전과 동일)
    except Exception as e:
        return f"최종 프롬프트 생성 중 오류: {e}"