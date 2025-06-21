import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text: str, max_tokens: int = 300) -> str:
    prompt = f"""
    다음 내용을 간결하고 핵심만 요약해 주세요:

    {text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
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
    prompt = f"""
    다음 요약문을 바탕으로 영상 내레이션용 스크립트를 작성해 주세요.
    감성적이고 자연스러운 이야기 형태로 만들어 주세요:

    {summary_text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
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
