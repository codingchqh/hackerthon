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


# --- 체인 1: 분석가를 위한 출력 구조 정의 (Pydantic) ---
class AnalysisResult(BaseModel):
    """육하원칙 분석 결과를 담는 데이터 구조"""
    is_complete: bool = Field(description="인터뷰 내용에 육하원칙의 핵심 요소(누가, 무엇을, 왜)가 모두 포함되어 있는지 여부")
    missing_elements: List[str] = Field(description="누락된 육하원칙 요소 목록 (한국어). 예: ['언제', '왜']")
    summary: str = Field(description="현재까지 파악된 인터뷰 내용의 간략한 요약")


# --- 체인 1: 분석가 체인 함수 ---
def analyze_transcript_for_completeness(transcript: str) -> AnalysisResult:
    """
    인터뷰 내용을 '창의적 추론 없이' 엄격하게 분석하여 완전성 여부를 진단하고,
    Pydantic 객체 형태로 구조화된 결과를 반환합니다.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    parser = PydanticOutputParser(pydantic_object=AnalysisResult)

    template = """
    You are a meticulous and strict analyst. Your task is to analyze an interview transcript based on the 5W1H principle (육하원칙) and determine if it contains enough information to create a video.

    A story is considered "complete" ONLY if it clearly contains the core elements: Who, What, and Why.
    The other elements (When, Where, How) are bonuses, but their absence must be noted in the 'missing_elements' list.

    CRITICAL INSTRUCTION: Do NOT infer, imagine, or create any information that is not explicitly present in the transcript. Your job is ONLY to analyze the provided text as-is.

    Transcript:
    ---
    {transcript}
    ---

    Based *only* on the transcript provided, provide your analysis in the following JSON format.
    If a core element (Who, What, Why) is missing, 'is_complete' must be false.
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        output = chain.invoke({"transcript": transcript})
        parsed_result = parser.parse(output['text'])
        return parsed_result
    except Exception as e:
        print(f"분석 체인 실행 중 오류: {e}")
        # 오류 발생 시, 일단은 '완성'으로 간주하여 무한 루프를 방지하거나,
        # 더 정교한 오류 처리를 추가할 수 있습니다.
        return AnalysisResult(is_complete=True, missing_elements=[], summary="분석 중 오류 발생")


# --- 체인 2: 친절한 인터뷰어 체인 함수 ---
def generate_follow_up_question(summary: str, missing_elements: List[str]) -> str:
    """
    분석 결과를 바탕으로 사용자에게 할 자연스러운 추가 질문을 생성합니다.
    """
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    missing_elements_str = ', '.join([f"'{el}'" for el in missing_elements])

    template = """
    You are a friendly and empathetic interviewer.
    A user has just shared a story, and you need to ask a follow-up question to get more details.

    Here is a summary of their story so far: "{summary}"

    Your analysis shows that the story is missing the following key elements to make a great video: {missing_elements_str}.

    Based on this, generate one friendly and encouraging follow-up question in Korean.
    Directly ask for the missing information, but in a natural, conversational way.
    For example, instead of "Tell me 'Why'", ask "그 이야기가 왜 그렇게 특별하게 느껴지셨는지 여쭤봐도 될까요?"

    Your follow-up question:
    """
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        result = chain.invoke({"summary": summary, "missing_elements_str": missing_elements_str})
        return result['text'].strip()
    except Exception as e:
        return f"추가 질문 생성 중 오류: {e}"


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


# --- 단독 실행 테스트 블록 ---
if __name__ == '__main__':
    print("--- gpt_summarizer.py 단독 테스트 시작 ---")

    # 1. 불완전한 텍스트 테스트
    print("\n[1] 불완전한 인터뷰 내용 분석")
    incomplete_transcript = "우리 아들이 상을 받았어요. 정말 자랑스러웠어요."
    print(f"입력 텍스트: {incomplete_transcript}")
    
    analysis = analyze_transcript_for_completeness(incomplete_transcript)
    print(f"분석 결과: 완전한가? -> {analysis.is_complete}")
    print(f"누락된 요소: {analysis.missing_elements}")
    print(f"내용 요약: {analysis.summary}")

    if not analysis.is_complete:
        print("\n[2] 추가 질문 생성")
        question = generate_follow_up_question(analysis.summary, analysis.missing_elements)
        print(f"생성된 추가 질문: {question}")

        # 2. 추가 답변을 포함한 완전한 텍스트 테스트
        print("\n[3] 완전한 인터뷰 내용으로 최종 프롬프트 생성")
        additional_answer = "작년 학교 축제에서 과학 경진대회 대상을 받았어요. 밤새 노력하는 걸 봐서 그런지 정말 특별하게 느껴졌어요."
        complete_transcript = incomplete_transcript + "\n" + additional_answer
        print(f"완성된 텍스트: {complete_transcript}")
        
        final_prompt = create_final_video_prompt(
            family_name="자랑스러운 우리집",
            theme="아이들의 사랑스러운 성장 기록",
            transcript=complete_transcript
        )
        print("\n--- 생성된 최종 영상 프롬프트 ---")
        print(final_prompt)