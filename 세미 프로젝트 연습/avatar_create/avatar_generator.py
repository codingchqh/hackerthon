import os
import requests
from PIL import Image
from io import BytesIO
import base64

# Runway API 키 (환경 변수 또는 직접 지정)
runway_api_key = os.getenv("RUNWAY_API_KEY")  # 또는 직접 입력: "your_runway_api_key_here"

# Runway API 기본 설정
RUNWAY_API_URL = "https://api.runwayml.com/v1/generate"
HEADERS = {
    "Authorization": f"Bearer {runway_api_key}",
    "Content-Type": "application/json"
}

def generate_avatar_image(prompt: str, n_images: int = 1, size: str = "512x512") -> list:
    """
    Runway Gen-2 기반으로 텍스트 프롬프트 기반 이미지 URL 생성 요청
    """
    if not runway_api_key:
        print("❌ Runway API 키가 설정되지 않았습니다.")
        return []

    payload = {
        "model": "gen2",
        "prompt": prompt,
        "n": n_images,
        "size": size
    }

    try:
        response = requests.post(RUNWAY_API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [item["url"] for item in data.get("output", [])]
    except requests.exceptions.RequestException as e:
        print(f"🌐 Runway API 요청 오류: {e}")
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
        print(f"⏱️ 타임아웃 발생: {url}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"🌐 네트워크 오류: {e}")
        raise
    except Exception as e:
        print(f"🖼️ 이미지 처리 오류: {e}")
        raise

def generate_avatar(prompt: str) -> Image.Image:
    """
    텍스트 프롬프트를 기반으로 아바타 이미지 생성 후 PIL 이미지로 반환
    """
    urls = generate_avatar_image(prompt=prompt, n_images=1)
    if not urls:
        raise ValueError("아바타 이미지 생성 실패: URL 반환 없음.")
    return download_image_from_url(urls[0])

# 테스트용 코드
if __name__ == "__main__":
    if not runway_api_key:
        print("❌ 테스트 실행 불가: RUNWAY_API_KEY 미설정.")
    else:
        test_prompt = "A realistic portrait of a Korean man smiling with soft lighting"
        print(f"테스트 프롬프트: {test_prompt}")

        urls = generate_avatar_image(test_prompt, n_images=1)
        if urls:
            print("✅ 생성된 이미지 URL:", urls[0])
            try:
                test_img = download_image_from_url(urls[0])
                test_img.show()
            except Exception as e:
                print(f"🖼️ 이미지 표시 오류: {e}")
        else:
            print("❌ 이미지 생성 실패.")
