import subprocess

# Ścieżki do plików
input_file = "/home/ai/s1/lab/BarPathAI/video/result3.mp4"
output_file = "/home/ai/s1/lab/BarPathAI/video/result3_rotated.mp4"

# Komenda FFmpeg do obrotu o 90 stopni w prawo (w prawo = clockwise)
command = [
    "ffmpeg",
    "-i", input_file,
    "-vf", "transpose=1",
    "-c:a", "copy",  # kopiowanie ścieżki audio bez zmian
    output_file
]

# Wykonanie komendy
subprocess.run(command)
print(f"Film został obrócony i zapisany jako {output_file}")
