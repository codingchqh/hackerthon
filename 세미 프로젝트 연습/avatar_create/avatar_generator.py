import os
import openai
import requests
from PIL import Image
from io import BytesIO

# 환경 변수에서 OpenAI API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = None
try:
    if openai_api_key:
        client = openai.OpenAI(api_key=openai_api_key)
    else:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
except openai.AuthenticationError as e:
    print(f"오류: OpenAI API 키 인증 실패: {e}")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 중 오류 발생: {e}")

def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    """
    DALL-E를 사용하여 프롬프트 기반 아바타 이미지 URL을 생성합니다.
    """
    if client is None:
        print("OpenAI 클라이언트가 초기화되지 않았습니다.")
        return []

    if not prompt:
        print("프롬프트가 비어 있습니다.")
        return []

    try:
        response = client.images.generate(
            model="dall-e-2",  # 또는 "dall-e-3" 사용 가능
            prompt=prompt,
            n=n_images,
            size=size,
            response_format="url"
        )
        image_urls = [item.url for item in response.data]
        return image_urls
    except openai.APIError as e:
        print(f"OpenAI API 오류: {e}")
        return []
    except Exception as e:
        print(f"이미지 생성 중 오류: {e}")
        return []

def download_image_from_url(url: str) -> Image.Image:
    """
    이미지 URL에서 PIL 이미지 객체 다운로드
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.Timeout:
        print(f"타임아웃 발생: {url}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"네트워크 오류: {e}")
        raise
    except Exception as e:
        print(f"이미지 처리 오류: {e}")
        raise

def generate_avatar(prompt: str) -> Image.Image:
    """
    텍스트 프롬프트를 기반으로 아바타 이미지 생성 후 PIL 이미지로 반환
    """
    urls = generate_avatar_image(prompt=prompt, n_images=1)
    if not urls:
        raise ValueError("아바타 이미지 생성 실패: URL 반환 없음.")
    return download_image_from_url(urls[0])

# 단독 실행 시 테스트 코드
if __name__ == "__main__":
    if not openai_api_key:
        print("테스트 실행 불가: OPENAI_API_KEY 미설정.")
    else:
        test_prompt = "A cute puppy wearing a superhero cape, digital art"
        print(f"테스트 프롬프트: {test_prompt}")

        urls = generate_avatar_image(test_prompt, n_images=1)
        if urls:
            print("생성된 이미지 URL:", urls[0])
            try:
                test_img = download_image_from_url(urls[0])
                test_img.show()
            except Exception as e:
                print(f"이미지 표시 오류: {e}")
        else:
            print("이미지 생성 실패.")
