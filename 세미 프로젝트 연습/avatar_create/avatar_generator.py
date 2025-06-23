import os
import openai # OpenAI 라이브러리 임포트
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv # .env 파일을 사용한다면 이 부분을 추가합니다.

# .env 파일을 사용한다면 이 부분을 추가합니다.
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# OpenAI API 키를 환경 변수에서 가져옵니다.
# .env 파일 사용 시 load_dotenv()가 먼저 실행되어야 합니다.
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
# API 키가 없으면 여기서 오류가 발생할 수 있습니다.
try:
    client = openai.OpenAI(api_key=openai_api_key)
except openai.AuthenticationError as e:
    print(f"OpenAI API 키를 로드하는 데 실패했습니다. 환경 변수(OPENAI_API_KEY)를 확인해주세요: {e}")
    # 프로그램 종료 또는 다른 처리
    exit()
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 중 예상치 못한 오류 발생: {e}")
    exit()


def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    """
    OpenAI DALL-E 모델을 사용하여 이미지를 생성합니다.
    """
    try:
        response = client.images.generate( # client.images.generate로 변경
            model="dall-e-2", # DALL-E 모델 지정 (dall-e-2 또는 dall-e-3)
            prompt=prompt,
            n=n_images,
            size=size,
            response_format="url" # URL 또는 b64_json을 선택할 수 있습니다.
        )
        # 응답 구조가 변경되었습니다.
        image_urls = [item.url for item in response.data]
        return image_urls
    except openai.APIError as e:
        print(f"OpenAI API 오류 발생: {e}")
        return []
    except Exception as e:
        print(f"이미지 생성 중 예상치 못한 오류 발생: {e}")
        return []

def download_image_from_url(url: str) -> Image.Image:
    """
    주어진 URL에서 이미지를 다운로드하여 PIL Image 객체로 반환합니다.
    """
    try:
        response = requests.get(url, stream=True) # stream=True 추가하여 큰 파일 처리 효율화
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 중 네트워크 오류 발생: {e}")
        raise # 예외를 다시 발생시켜 상위 호출자에게 알림
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        raise # 예외를 다시 발생시켜 상위 호출자에게 알림

if __name__ == "__main__":
    prompt_text = "A beautiful, friendly cartoon avatar of a young woman, digital art style"
    
    # generate_avatar_image 함수가 오류를 반환할 수 있으므로,
    # 함수 호출 전에 API 키가 유효한지 확인하는 것이 좋습니다.
    if not openai_api_key:
        print("OPENAI_API_KEY가 설정되지 않았습니다. 환경 변수를 확인해주세요.")
    else:
        urls = generate_avatar_image(prompt_text)
        if urls:
            print("생성된 이미지 URL:", urls[0])
            try:
                avatar_img = download_image_from_url(urls[0])
                avatar_img.show()
                print("이미지가 성공적으로 표시되었습니다.")
            except Exception as e:
                print(f"이미지 표시 중 오류 발생: {e}")
        else:
            print("이미지 생성 실패 또는 URL이 반환되지 않았습니다.")