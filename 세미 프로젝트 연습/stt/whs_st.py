import sounddevice as sd # 사운드 장치 제어를 위한 라이브러리 (녹음, 재생 등)
import numpy as np # 숫자 계산, 특히 배열 처리를 위한 라이브러리
import wave # WAV 오디오 파일 읽기/쓰기를 위한 라이브러리
import io # 인메모리 파일 작업을 위한 라이브러리 (바이트 스트림 처리)
import whisper # 음성-텍스트 변환 (STT)을 위한 OpenAI의 Whisper 모델 라이브러리
import tempfile # 임시 파일 생성을 위한 라이브러리
import os # 운영체제와 상호작용하기 위한 라이브러리 (파일 시스템 접근 등)
from summarizer.gpt_summarizer import summarize_text, generate_video_script  # 'summarizer' 패키지에서 텍스트 요약 및 비디오 스크립트 생성 함수 임포트

# Whisper 모델 로드 함수
def load_model(model_name="base", device=None):
    """
    지정된 이름의 Whisper 모델을 로드합니다.
    Args:
        model_name (str): 로드할 Whisper 모델의 이름 (예: "base", "small", "medium", "large"). 기본값은 "base".
        device (str, optional): 모델을 로드할 장치 ("cpu" 또는 "cuda"). None이면 자동으로 결정.
    Returns:
        whisper.model: 로드된 Whisper 모델 객체.
    """
    return whisper.load_model(model_name, device=device)

# 오디오 녹음 함수 (장치 오류 방지용 수정 포함)
def record_audio(duration_sec=5, fs=16000, device=None):
    """
    지정된 시간 동안 오디오를 녹음합니다.
    Args:
        duration_sec (int): 녹음할 시간 (초). 기본값은 5초.
        fs (int): 샘플링 주파수 (초당 샘플 수). 기본값은 16000 Hz.
        device (int, optional): 사용할 오디오 입력 장치의 번호. None이면 기본 장치 사용 시도.
    Returns:
        numpy.ndarray: 녹음된 오디오 데이터 (평탄화된 NumPy 배열).
    Raises:
        RuntimeError: 기본 입력 장치가 없거나 유효하지 않을 경우 발생.
    """
    if device is None:
        # 기본 입력 장치 인덱스를 가져옴
        device = sd.default.device[0]
        # 기본 장치가 없거나 유효하지 않은 경우 오류 발생
        if device is None or device < 0:
            raise RuntimeError("기본 입력 장치가 없습니다. device 파라미터로 입력 장치 번호를 지정해 주세요.")
    print(f"녹음에 사용할 입력 장치 번호: {device}")
    # 오디오 녹음을 시작
    # int(duration_sec * fs): 녹음할 샘플 수 계산
    # samplerate=fs: 샘플링 주파수 설정
    # channels=1: 모노 채널로 녹음
    # dtype='int16': 16비트 정수형 데이터 타입으로 녹음
    # device=device: 사용할 오디오 입력 장치 지정
    audio = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16', device=device)
    # 녹음이 완료될 때까지 대기
    sd.wait()
    # 녹음된 오디오 데이터를 1차원 배열로 평탄화하여 반환
    return audio.flatten()

# numpy 배열 → WAV 형식 메모리 스트림 변환 함수
def numpy_to_wav_bytes(audio_np, fs=16000):
    """
    NumPy 배열 형태의 오디오 데이터를 WAV 형식의 바이트 스트림으로 변환합니다.
    Args:
        audio_np (numpy.ndarray): 오디오 데이터가 담긴 NumPy 배열.
        fs (int): 샘플링 주파수. 기본값은 16000 Hz.
    Returns:
        io.BytesIO: WAV 형식의 오디오 데이터가 담긴 메모리 스트림.
    """
    # 데이터를 저장할 인메모리 바이트 버퍼 생성
    buffer = io.BytesIO()
    # wave 모듈을 사용하여 버퍼에 WAV 파일 형식으로 쓰기
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1) # 채널 수 설정 (모노)
        wf.setsampwidth(2) # 샘플 너비 설정 (2바이트 = 16비트)
        wf.setframerate(fs) # 프레임 레이트 (샘플링 주파수) 설정
        wf.writeframes(audio_np.tobytes()) # NumPy 배열을 바이트로 변환하여 WAV 프레임으로 기록
    # 버퍼의 읽기/쓰기 위치를 처음으로 되돌림 (다른 함수가 읽을 수 있도록)
    buffer.seek(0)
    return buffer

# WAV BytesIO → Whisper 전사 + 요약 + 프롬프트 생성 함수
def transcribe_audio(model, wav_io, return_all=False):
    """
    WAV 바이트 스트림을 Whisper 모델로 전사(transcribe)하고,
    전사된 텍스트를 요약한 다음, 요약본으로 비디오 스크립트 프롬프트를 생성합니다.
    Args:
        model (whisper.model): 로드된 Whisper 모델 객체.
        wav_io (io.BytesIO): WAV 형식의 오디오 데이터가 담긴 메모리 스트림.
        return_all (bool): True이면 전사, 요약, 비디오 스크립트 모두를 딕셔너리 형태로 반환.
                           False이면 전사된 텍스트만 반환. 기본값은 False.
    Returns:
        Union[str, dict]: 전사된 텍스트 또는 전사, 요약, 스크립트가 포함된 딕셔너리.
    """
    # 임시 WAV 파일 생성 (Whisper 모델이 파일 경로를 인자로 받기 때문)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(wav_io.read()) # 메모리 스트림의 내용을 임시 파일에 기록
        tmp_path = tmp_file.name # 임시 파일의 경로 저장

    # Whisper 모델을 사용하여 임시 파일의 오디오를 텍스트로 전사
    result = model.transcribe(tmp_path)
    # 사용이 끝난 임시 파일 삭제
    os.remove(tmp_path)

    # 전사된 텍스트 추출
    transcript = result["text"]
    # 전사된 텍스트를 요약
    summary = summarize_text(transcript)
    # 요약된 텍스트를 바탕으로 비디오 스크립트 프롬프트 생성
    script = generate_video_script(summary)

    # return_all이 True이면 모든 결과(전사, 요약, 스크립트)를 딕셔너리 형태로 반환
    if return_all:
        return {
            "transcript": transcript,
            "summary": summary,
            "video_script": script
        }
    # return_all이 False이면 전사된 텍스트만 반환
    return transcript