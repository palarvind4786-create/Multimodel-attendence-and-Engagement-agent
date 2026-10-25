import time
import os
import tempfile
import psutil
import torch
import cv2
import numpy as np
from ultralytics import YOLO

def create_sample_video(path, frames=60, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 640, 480
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), 220, dtype=np.uint8)
        # Draw moving circle/rectangle simulating person
        x = 100 + (i * 4) % 300
        cv2.rectangle(frame, (x, 150), (x + 80, 400), (60, 60, 60), -1)
        cv2.circle(frame, (x + 40, 100), 40, (180, 140, 120), -1)
        out.write(frame)
    out.release()

def run_performance_benchmark():
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)
    cpu_before = psutil.cpu_percent(interval=0.2)

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"
    print(f"=== HARDWARE & ENVIRONMENT ===")
    print(f"CUDA Available: {has_cuda}")
    print(f"GPU Device: {device_name}")
    print(f"Baseline RAM: {ram_before_mb:.1f} MB | Initial CPU: {cpu_before:.1f}%")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = tf.name

    try:
        frames_count = 60
        create_sample_video(video_path, frames=frames_count, fps=20)
        file_size_kb = os.path.getsize(video_path) / 1024
        print(f"Generated test video: {frames_count} frames ({file_size_kb:.1f} KB)")

        # 1. GPU Benchmark (if CUDA)
        if has_cuda:
            torch.cuda.empty_cache()
            gpu_mem_start = torch.cuda.memory_allocated(0) / (1024 * 1024)
            model_gpu = YOLO("yolov8n.pt")
            
            t0 = time.perf_counter()
            cap = cv2.VideoCapture(video_path)
            gpu_processed = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _ = model_gpu(frame, device=0, verbose=False)
                gpu_processed += 1
            cap.release()
            t1 = time.perf_counter()
            gpu_duration = t1 - t0
            gpu_fps = gpu_processed / gpu_duration if gpu_duration > 0 else 0
            gpu_mem_peak = torch.cuda.max_memory_allocated(0) / (1024 * 1024)

            print(f"\n=== GPU ACCELERATION INFERENCE (CUDA:0) ===")
            print(f"Processed: {gpu_processed} frames in {gpu_duration:.2f}s ({gpu_fps:.1f} FPS)")
            print(f"GPU VRAM Active: {gpu_mem_peak:.1f} MB (Delta: {gpu_mem_peak - gpu_mem_start:.1f} MB)")

        # 2. CPU Fallback Benchmark
        model_cpu = YOLO("yolov8n.pt")
        t0_cpu = time.perf_counter()
        cap_cpu = cv2.VideoCapture(video_path)
        cpu_processed = 0
        while cap_cpu.isOpened():
            ret, frame = cap_cpu.read()
            if not ret:
                break
            _ = model_cpu(frame, device='cpu', verbose=False)
            cpu_processed += 1
        cap_cpu.release()
        t1_cpu = time.perf_counter()
        cpu_duration = t1_cpu - t0_cpu
        cpu_fps = cpu_processed / cpu_duration if cpu_duration > 0 else 0

        ram_after_mb = process.memory_info().rss / (1024 * 1024)
        cpu_after = psutil.cpu_percent(interval=0.2)

        print(f"\n=== CPU FALLBACK INFERENCE (CPU) ===")
        print(f"Processed: {cpu_processed} frames in {cpu_duration:.2f}s ({cpu_fps:.1f} FPS)")
        print(f"RAM Usage: {ram_after_mb:.1f} MB (Delta: {ram_after_mb - ram_before_mb:.1f} MB)")
        print(f"CPU Utilization: {cpu_after:.1f}%")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"\nEphemeral test video cleaned up: {not os.path.exists(video_path)}")

if __name__ == "__main__":
    run_performance_benchmark()
