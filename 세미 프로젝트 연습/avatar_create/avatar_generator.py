import os
import openai
import requests
from PIL import Image
from io import BytesIO

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n_images,
            size=size
        )
        image_urls = [item['url'] for item in response['data']]
        return image_urls
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        return []

def download_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img

if __name__ == "__main__":
    prompt_text = "A beautiful, friendly cartoon avatar of a young woman, digital art style"
    urls = generate_avatar_image(prompt_text)
    if urls:
        print("생성된 이미지 URL:", urls[0])
        avatar_img = download_image_from_url(urls[0])
        avatar_img.show()
    else:
        print("이미지 생성 실패")
