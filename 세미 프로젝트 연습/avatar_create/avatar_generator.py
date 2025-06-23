import os
import openai
import requests
from PIL import Image
from io import BytesIO

# OpenAI API 키 환경변수에서 로드 (필요시 직접 입력 가능)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    """
    OpenAI 이미지 생성 API를 호출하여 아바타 이미지를 생성하고, 이미지 URL 리스트를 반환합니다.
    """
    try:
        response = openai.images.generate(
            prompt=prompt,
            n=n_images,
            size=size
        )
        # response.data는 생성된 이미지 URL 목록
        image_urls = [item.url for item in response.data]
        return image_urls
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        return []

def download_image_from_url(url: str) -> Image.Image:
    """
    URL에서 이미지를 다운로드하여 PIL 이미지 객체로 반환
    """
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

# 사용 예시
if __name__ == "__main__":
    prompt_text = "A beautiful, friendly cartoon avatar of a young woman, digital art style"
    urls = generate_avatar_image(prompt_text)
    if urls:
        print("생성된 이미지 URL:", urls[0])
        avatar_img = download_image_from_url(urls[0])
        avatar_img.show()  # 이미지 표시 (로컬 환경에서만)
    else:
        print("이미지 생성 실패")
