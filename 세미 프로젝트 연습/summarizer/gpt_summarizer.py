import os
import openai

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text: str, max_tokens: int = 300) -> str:
    """
    주어진 텍스트를 간결한 영어로 요약합니다.
    """
    # 프롬프트를 영어로 변경
    prompt = f"""
    Please summarize the following content concisely, focusing on the key points:

    {text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # 시스템 메시지도 영어로 변경
            {"role": "system", "content": "You are a helpful summarization expert."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.5,
        n=1,
    )
    summary = response.choices[0].message.content.strip()
    return summary


def generate_video_script(summary_text: str, max_tokens: int = 50) -> str:
    """
    요약문을 바탕으로 6초 분량의 짧은 영어 영상 스크립트를 생성합니다.
    """
    # 프롬프트를 영어로 변경하고, 길이에 대한 명확한 지시사항 추가
    prompt = f"""
    Based on the following summary, create a narration script for a short video.
    The script MUST be very brief, suitable for a video of about 6 seconds (approximately 15-25 words).
    Make it emotional and natural:

    {summary_text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # 시스템 메시지도 영어로 변경
            {"role": "system", "content": "You are an emotional and creative video scriptwriter."},
            {"role": "user", "content": prompt},
        ],
        # max_tokens 값을 6초 분량에 맞게 크게 줄여서 불필요한 생성을 방지
        max_tokens=max_tokens,
        temperature=0.7,
        n=1,
    )
    script = response.choices[0].message.content.strip()
    return script

# --- 테스트용 코드 ---
if __name__ == '__main__':
    # 긴 원본 텍스트 예시 (영문)
    sample_text = """
    Gemini is a family of multimodal large language models developed by Google.
    Announced on December 6, 2023, it is positioned as a competitor to OpenAI's GPT-4.
    Gemini 1.0 was released in three sizes: Ultra, Pro, and Nano.
    Google states that Gemini can understand and respond to text, images, audio, and video,
    making it a powerful tool for a wide range of applications. It has been integrated into
    various Google products, including the chatbot formerly known as Bard, which was rebranded as Gemini.
    """

    print("--- 1. Summarizing Text ---")
    summary = summarize_text(sample_text)
    print("Summary:", summary)
    print("\n" + "="*30 + "\n")

    print("--- 2. Generating 6-second Video Script ---")
    video_script = generate_video_script(summary)
    print("Video Script:", video_script)