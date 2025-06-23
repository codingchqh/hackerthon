# avatar_create/avatar_generator.py

import os
import openai
import requests
from PIL import Image
from io import BytesIO

# .env 파일 로드 부분을 제거합니다.
# from dotenv import load_dotenv # 이 줄은 이제 필요 없습니다.

# OpenAI API 키를 환경 변수에서 가져옵니다.
# 이제 이 스크립트가 실행되는 환경(운영 체제)에 OPENAI_API_KEY가 설정되어 있어야 합니다.
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = None # 초기화
try:
    if openai_api_key:
        client = openai.OpenAI(api_key=openai_api_key)
    else:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. API 호출이 실패할 수 있습니다.")
        # 클라이언트가 초기화되지 않은 상태로 유지됨
except openai.AuthenticationError as e:
    print(f"오류: OpenAI API 키 인증에 실패했습니다. 환경 변수 OPENAI_API_KEY를 확인해주세요: {e}")
    client = None # 인증 실패 시 클라이언트 초기화 안함
except Exception as e:
    print(f"오류: OpenAI 클라이언트 초기화 중 예상치 못한 오류 발생: {e}")
    client = None

def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    """
    OpenAI DALL-E 모델을 사용하여 이미지를 생성합니다.
    주의: 이 함수는 텍스트 프롬프트만 받습니다. face_img와 같은 이미지 객체를 직접 받지 않습니다.
    """
    if client is None:
        print("오류: OpenAI 클라이언트가 초기화되지 않았습니다. API 키 설정을 확인해주세요.")
        return []

    if not prompt:
        print("오류: 이미지 생성을 위한 프롬프트가 비어 있습니다.")
        return []

    try:
        response = client.images.generate(
            model="dall-e-2", # 또는 "dall-e-3"
            prompt=prompt,
            n=n_images,
            size=size,
            response_format="url"
        )
        image_urls = [item.url for item in response.data]
        return image_urls
    except openai.APIError as e:
        print(f"OpenAI API 오류 발생 (generate_avatar_image): {e}")
        return []
    except Exception as e:
        print(f"이미지 생성 중 예상치 못한 오류 발생 (generate_avatar_image): {e}")
        return []

def download_image_from_url(url: str) -> Image.Image:
    """
    주어진 URL에서 이미지를 다운로드하여 PIL Image 객체로 반환합니다.
    """
    try:
        response = requests.get(url, stream=True, timeout=10) # 타임아웃 추가
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.Timeout:
        print(f"이미지 다운로드 타임아웃 발생: {url}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 중 네트워크 오류 발생: {url}, 오류: {e}")
        raise
    except Exception as e:
        print(f"다운로드된 이미지 처리 중 오류 발생: {url}, 오류: {e}")
        raise

if __name__ == "__main__":
    # 이 파일 단독 실행 테스트 (API 키가 운영 체제 환경 변수에 설정되어 있어야 함)
    if not openai_api_key:
        print("테스트 실행 불가: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    else:
        test_prompt = "A cute puppy wearing a superhero cape, digital art"
        print(f"테스트 프롬프트: {test_prompt}")
        
        urls = generate_avatar_image(test_prompt, n_images=1)
        if urls:
            print("생성된 이미지 URL:", urls[0])
            try:
                test_img = download_image_from_url(urls[0])
                test_img.show()
                print("테스트 이미지가 성공적으로 표시되었습니다.")
            except Exception as e:
                print(f"테스트 이미지 표시 중 오류 발생: {e}")
        else:
            print("테스트 이미지 생성 실패 또는 URL이 반환되지 않았습니다.")