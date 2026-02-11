import os
import numpy as np
import subprocess
import cv2
import shutil
from PIL import Image
import librosa
import soundfile as sf
import math
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.realpath(__file__))

def split_long_video(input_video_path, output_dir, segment_duration=20):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_video_path))[0]  # e.g. 0001
   
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", input_video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    total_duration = round(float(result.stdout.strip()))
    n_segments = math.ceil(total_duration / segment_duration)

    for i in range(n_segments):
        start_time = i * segment_duration
        output_name = f"{filename}_{i+1:02d}.mp4"
        output_path = os.path.join(output_dir, output_name)

        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), 
            "-i", input_video_path,
            "-t", str(segment_duration),
            "-c", "copy", 
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # print(f"✅ Split {filename} into {n_segments} parts.")

def truncate(audio_array: np.ndarray, sr: int, annot_data: np.ndarray, fps: int):
        audio_duration = len(audio_array) / sr
        annot_duration = len(annot_data) / fps
        duration = min(audio_duration, annot_duration, 20)
        audio_length = round(duration * sr)
        annot_length = round(duration * fps)
        return audio_array[:audio_length], annot_data[:annot_length]

def get_audio_frames(input_video_path, output_dir, name, sample_rate=16000, channel=1, fps=25):
    audio_path = os.path.join(output_dir, f"{name}.wav")
    audio_cmd = [
        "ffmpeg", "-y", "-i", input_video_path,
        "-vn", "-ar", str(sample_rate), "-ac", str(channel),
        "-acodec", "pcm_s16le",
        audio_path
    ]
    subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"{input_video_path} decouping finished!")

if __name__ =="__main__":
    # set higher sr for audio
    video_dir = os.path.join(cur_dir, "test_data", "test_videos")
    mp4_files = sorted(os.listdir(video_dir))
    if '.git' in mp4_files:
        mp4_files.remove('.git')
    # print(mp4_files)

    # for mp4_file in tqdm(mp4_files):
    #     input_file_path = os.path.join(video_dir, mp4_file)
    #     split_long_video(input_video_path=input_file_path, output_dir=split_long_dir, segment_duration=20)

    for mp4_file in tqdm(mp4_files):
        input_file_path = os.path.join(video_dir, mp4_file)
        name = mp4_file.replace(".mp4", "")
        output_dir = os.path.join(cur_dir, "test_data", "test_audios")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        get_audio_frames(input_video_path=input_file_path, output_dir=output_dir, name=name, sample_rate=16000, channel=1, fps=25)

