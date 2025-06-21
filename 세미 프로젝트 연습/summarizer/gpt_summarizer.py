# gpt_summarizer.py
import os
import openai

# OpenAI API 키 설정 (환경변수 OPENAI_API_KEY를 사용)
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text: str, max_tokens: int = 300) -> str:
    """
    긴 텍스트를 간결하게 요약하는 함수
    """
    prompt = f"""
    다음 내용을 간결하고 핵심만 요약해 주세요:

    {text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 사용 가능한 모델명으로 변경하세요
        messages=[
            {"role": "system", "content": "당신은 친절한 요약 전문가입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.5,
        n=1,
    )

    summary = response.choices[0].message.content.strip()
    return summary


def generate_video_script(summary_text: str, max_tokens: int = 500) -> str:
    """
    요약문을 바탕으로 감성적이고 영상 내레이션에 적합한 스크립트(프로포트) 생성
    """
    prompt = f"""
    다음 요약문을 바탕으로 영상 내레이션용 스크립트를 작성해 주세요.
    감성적이고 자연스러운 이야기 형태로 만들어 주세요:

    {summary_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # 사용 가능한 모델명으로 변경하세요
        messages=[
            {"role": "system", "content": "당신은 감성적이고 창의적인 영상 스크립트 작가입니다."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        n=1,
    )

    script = response.choices[0].message.content.strip()
    return script


# 테스트용 main
if __name__ == "__main__":
    sample_text = """
    여기에 whs_st.py에서 받아온 긴 텍스트 데이터를 넣고 실행해보세요.
    """
    summary = summarize_text(sample_text)
    print("==== 요약문 ====")
    print(summary)

    video_script = generate_video_script(summary)
    print("\n==== 영상 스크립트 ====")
    print(video_script)
