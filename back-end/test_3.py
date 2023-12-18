import subprocess

def split_video(input_file, output_file1, output_file2, duration1):
    # FFmpeg command to split the video
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-t', duration1,
        '-c', 'copy',
        output_file1,
        '-ss', duration1,
        '-c', 'copy',
        output_file2
    ]

    # Run the FFmpeg command
    subprocess.run(cmd)

if __name__ == "__main__":
    # Replace these with your input and output file paths
    input_file = r"C:\Users\Admin\Downloads\video.mp4"
    output_file1 = 'output_part1.mp4'
    output_file2 = 'output_part2.mp4'

    # Specify the duration for the first part (in format HH:MM:SS)
    duration1 = '00:02:30'

    # Call the function to split the video
    split_video(input_file, output_file1, output_file2, duration1)
