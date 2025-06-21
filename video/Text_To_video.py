from moviepy.editor import TextClip, CompositeVideoClip # MoviePy 라이브러리 임포트: 비디오 편집 (텍스트 클립, 클립 합성)
import numpy as np # NumPy 라이브러리 임포트: 숫자 배열 처리에 사용
from scipy.io import wavfile # SciPy 라이브러리에서 wavfile 모듈 임포트: WAV 파일 읽기/쓰기에 사용

def make_video_from_text(text: str, output_path="output.mp4", duration=10):
    """
    텍스트로부터 간단한 비디오 클립을 생성합니다.
    Args:
        text (str): 비디오에 표시될 텍스트.
        output_path (str): 생성될 비디오 파일의 저장 경로. 기본값은 "output.mp4".
        duration (int): 비디오 클립의 길이 (초). 기본값은 10초.
    """
    # TextClip 생성: 텍스트, 폰트 크기, 색상, 비디오 해상도, 폰트, 배경색 지정
    # 'Malgun Gothic' 폰트는 한글을 표시하기 위해 지정됨
    clip = TextClip(text, fontsize=32, color='white', size=(1280, 720), font='Malgun Gothic', bg_color='black')
    # 클립의 재생 시간 설정
    clip = clip.set_duration(duration)
    # 텍스트 클립으로 구성된 비디오 클립 생성 (단일 클립이므로 CompositeVideoClip에 리스트로 전달)
    video = CompositeVideoClip([clip])
    # 비디오 파일을 지정된 경로에 저장, 프레임 속도(fps)는 24로 설정
    video.write_videofile(output_path, fps=24)

def save_wav_file(path, audio_np, fs=16000):
    """
    NumPy 배열 형태의 오디오 데이터를 WAV 파일로 저장합니다.
    Args:
        path (str): WAV 파일을 저장할 경로.
        audio_np (numpy.ndarray): 저장할 오디오 데이터 (NumPy 배열).
        fs (int): 오디오의 샘플링 주파수. 기본값은 16000 Hz.
    """
    # scipy.io.wavfile.write 함수를 사용하여 WAV 파일로 데이터 쓰기
    # path: 저장할 파일 경로
    # fs: 샘플링 주파수
    # audio_np: 오디오 데이터 배열
    wavfile.write(path, fs, audio_np)