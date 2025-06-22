import os
import io
from PIL import Image
from stt.whs_st import load_model, record_audio, numpy_to_wav_bytes, transcribe_audio
from summarizer.gpt_summarizer import summarize_text, generate_video_script
from camera.face_capture import extract_face
from avatar_create.avatar_generator import generate_avatar  # ✅ 아바타 생성 함수 import
from video.Text_To_video import make_video_from_text, save_wav_file

# --- 설정 ---
AUDIO_FILE = "audio/recorded.wav"
ORIGINAL_IMAGE_FILE = "camera/sample.jpg"
FACE_IMAGE_FILE = "camera/face_extracted.jpg"
AVATAR_IMAGE_FILE = "camera/avatar.jpg"  # ✅ 아바타 이미지 경로 추가
VIDEO_OUTPUT_PATH = "output/output.mp4"

# --- 디렉토리 생성 ---
os.makedirs("audio", exist_ok=True)
os.makedirs("camera", exist_ok=True)
os.makedirs("output", exist_ok=True)

# --- 1. 오디오 녹음 및 텍스트 전사 ---
print("1. 오디오 녹음 및 전사 시작...")
model = load_model()
audio_np = record_audio(duration_sec=5)
wav_bytes = numpy_to_wav_bytes(audio_np)

# WAV 저장
with open(AUDIO_FILE, "wb") as f:
    f.write(wav_bytes.read())
print(f"녹음된 오디오 저장 완료: {AUDIO_FILE}")

# 전사
with open(AUDIO_FILE, "rb") as f:
    transcription, summary, script = transcribe_audio(model, io.BytesIO(f.read()), return_all=True).values()

print("\n📝 전사 결과:")
print(transcription)
print("\n📑 요약:")
print(summary)
print("\n🎬 영상 스크립트:")
print(script)

# --- 2. 얼굴 이미지 처리 ---
print("\n2. 얼굴 이미지 추출 시작...")
if not os.path.exists(ORIGINAL_IMAGE_FILE):
    raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {ORIGINAL_IMAGE_FILE}")

original_img = Image.open(ORIGINAL_IMAGE_FILE)
face_img = extract_face(original_img, save_path=FACE_IMAGE_FILE)
if face_img is None:
    print("⚠️ 얼굴을 찾지 못했습니다. 원본 이미지를 사용합니다.")
    image_path = ORIGINAL_IMAGE_FILE
else:
    print(f"✅ 얼굴 이미지 저장 완료: {FACE_IMAGE_FILE}")
    
    # --- 2.5 아바타 이미지 생성 ---
    print("\n2.5 아바타 이미지 생성 시작...")
    avatar_img = generate_avatar(face_img)
    avatar_img.save(AVATAR_IMAGE_FILE)
    print(f"✅ 아바타 이미지 저장 완료: {AVATAR_IMAGE_FILE}")
    
    image_path = AVATAR_IMAGE_FILE  # ✅ 영상 생성에 사용할 이미지 경로 갱신

# --- 3. 영상 생성 ---
print("\n3. 영상 생성 시작...")
make_video_from_text(script, output_path=VIDEO_OUTPUT_PATH, image_path=image_path)
print(f"\n✅ 영상 생성 완료: {VIDEO_OUTPUT_PATH}")
