import io # 인메모리 파일(바이트 스트림)을 다루는 데 사용됩니다.
from PIL import Image # Pillow 라이브러리로 이미지를 열고 처리하는 데 사용됩니다.

# 사용자 정의 모듈에서 필요한 함수들을 임포트합니다.
from stt.whs_st import load_model, record_audio, numpy_to_wav_bytes, transcribe_audio # 음성-텍스트 변환(STT) 관련 함수들
from summarizer.gpt_summarizer import summarize_text, generate_video_script # 텍스트 요약 및 비디오 스크립트 생성 함수들
from camera.face_capture import extract_face # 이미지에서 얼굴을 추출하는 함수
from video.Text_To_video import create_video  # 텍스트와 이미지로 비디오를 생성하는 함수 (직접 구현 필요)

import os # 운영체제와 상호작용하여 파일 경로 등을 다루는 데 사용됩니다.

# 파일 경로 상수 정의
AUDIO_FILE = "audio/recorded.wav" # 녹음된 오디오를 저장할 WAV 파일 경로
ORIGINAL_IMAGE_FILE = "camera/sample.jpg" # 원본 이미지 파일 경로
FACE_IMAGE_FILE = "camera/face_extracted.jpg"  # 추출된 얼굴 이미지를 저장할 경로

# --- 프로세스 시작 ---

# 1. 음성 녹음 → 텍스트 전사
print("1. 음성 녹음 및 텍스트 전사 시작...")
model = load_model() # Whisper 모델을 로드합니다. (캐시 기능이 있다면 재사용)
audio_np = record_audio(duration_sec=5) # 5초 동안 오디오를 녹음하고 NumPy 배열로 받습니다.
wav_bytes = numpy_to_wav_bytes(audio_np) # 녹음된 NumPy 배열을 WAV 형식의 바이트 스트림으로 변환합니다.

# 변환된 WAV 바이트 스트림을 실제 파일로 저장합니다. (나중에 다시 읽을 수 있도록)
with open(AUDIO_FILE, "wb") as f:
    f.write(wav_bytes.read())
print(f"녹음된 오디오를 저장했습니다: {AUDIO_FILE}")

# 저장된 오디오 파일을 다시 읽어와 Whisper 모델로 텍스트 전사를 수행합니다.
# io.BytesIO(open(AUDIO_FILE, "rb").read()): 파일을 바이트 스트림으로 읽어서 transcribe_audio 함수에 전달합니다.
transcription = transcribe_audio(model, io.BytesIO(open(AUDIO_FILE, "rb").read()))
print("\n📝 전사 결과:")
print(transcription)

# 2. 텍스트 요약 → 영상 스크립트 생성
print("\n2. 텍스트 요약 및 영상 스크립트 생성 시작...")
summary = summarize_text(transcription) # 전사된 텍스트를 요약합니다.
script = generate_video_script(summary) # 요약된 텍스트를 바탕으로 영상 스크립트를 생성합니다.
print("📜 생성된 영상 스크립트:")
print(script)

# 3. 얼굴 추출 후 저장
print("\n3. 얼굴 추출 및 저장 시작...")
image = Image.open(ORIGINAL_IMAGE_FILE) # 지정된 원본 이미지 파일을 엽니다.
# 이미지에서 얼굴을 추출하고, 추출된 얼굴 이미지를 FACE_IMAGE_FILE 경로에 저장합니다.
# (extract_face 함수에 save_path 인자 처리가 구현되어 있어야 함)
face_img = extract_face(image, save_path=FACE_IMAGE_FILE)

if face_img is None: # 얼굴 추출에 실패한 경우
    print("얼굴을 찾지 못했습니다. 원본 이미지를 사용하여 비디오를 생성합니다.")
    image_path_for_video = ORIGINAL_IMAGE_FILE # 비디오 생성 시 원본 이미지를 사용하도록 경로를 설정합니다.
else: # 얼굴 추출에 성공한 경우
    print(f"얼굴 이미지를 저장했습니다: {FACE_IMAGE_FILE}")
    image_path_for_video = FACE_IMAGE_FILE # 비디오 생성 시 추출된 얼굴 이미지를 사용하도록 경로를 설정합니다.

# 4. 영상 생성 (스크립트 + 얼굴 이미지 + TTS)
print("\n4. 영상 생성 시작...")
# 생성된 스크립트 텍스트와 사용할 이미지 경로를 기반으로 최종 비디오를 생성합니다.
# 이 함수 내부에는 텍스트-음성 변환(TTS) 및 비디오 합성 로직이 포함되어야 합니다.
create_video(script_text=script, image_path=image_path_for_video, output_path="output.mp4")

print("\n✅ 영상 생성 완료 → output.mp4") # 모든 프로세스가 완료되었음을 알리는 메시지입니다.