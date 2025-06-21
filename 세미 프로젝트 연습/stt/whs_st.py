import numpy as np
import wave
import io
import whisper
import tempfile
import os
import platform
from summarizer.gpt_summarizer import summarize_text, generate_video_script

# 플랫폼 확인
IS_LOCAL = platform.system() != "Linux"  # 로컬이면 True, 클라우드(Linux)면 False

if IS_LOCAL:
    import sounddevice as sd

def load_model(model_name="base", device=None):
    return whisper.load_model(model_name, device=device)

# 로컬 전용 오디오 녹음 함수
def record_audio(duration_sec=5, fs=16000, device=None):
    if not IS_LOCAL:
        raise RuntimeError("이 함수는 로컬 환경에서만 사용할 수 있습니다.")
    
    if device is None:
        device = sd.default.device[0]
        if device is None or device < 0:
            raise RuntimeError("기본 입력 장치가 없습니다. device 파라미터로 입력 장치 번호를 지정해 주세요.")
    
    print(f"녹음에 사용할 입력 장치 번호: {device}")
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    sd.wait()
    return audio.flatten()

def numpy_to_wav_bytes(audio_np, fs=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer

def transcribe_audio(model, wav_io, return_all=False):
    # wav_io가 파일 업로드(UploadedFile) 또는 BytesIO 둘 다 지원하도록
    if hasattr(wav_io, "read"):
        # 파일이나 BytesIO이면 그대로 읽음
        content = wav_io.read()
    else:
        raise ValueError("지원하지 않는 오디오 입력 타입입니다.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name

    result = model.transcribe(tmp_path)
    os.remove(tmp_path)

    transcript = result["text"]
    summary = summarize_text(transcript)
    script = generate_video_script(summary)

    if return_all:
        return {
            "transcript": transcript,
            "summary": summary,
            "video_script": script
        }

    return transcript
