import os
import glob
import argparse
import cv2
import numpy as np
from decord import VideoReader


def split_video(input_video_path, output_folder, frame_count=16, prefix=""):
    """
    将一个视频文件切分为多个小视频文件，每个视频文件包含指定数量的帧。

    :param input_video_path: 输入视频文件的路径
    :param output_folder: 输出视频文件的目录
    :param frame_count: 每个小视频包含的帧数
    :param prefix: 输出视频文件的前缀
    """
    # 创建输出目录，如果不存在
    os.makedirs(output_folder, exist_ok=True)

    # 读取视频
    vr = VideoReader(input_video_path)

    # 获取视频总帧数
    total_frames = len(vr)
    print(f"Total frames in the video: {total_frames}")

    # 计算每个切片的起始和结束索引
    stride = frame_count
    for i in range(0, total_frames, stride):
        end = min(i + stride, total_frames)
        if end - i < frame_count:
            break

        # 提取帧
        frames = vr[i:end].asnumpy()

        # RGB to BGR
        frames = frames[..., ::-1]

        # 创建输出视频文件名
        base_name = os.path.basename(input_video_path).replace('.mp4', '')  # 去掉扩展名
        output_video_path = os.path.join(output_folder, f"{prefix}{base_name}_{i // stride}.mp4")

        # 保存视频片段
        save_video_from_frames(frames, output_video_path)

        print(f"Split video saved at: {output_video_path}")

def save_video_from_frames(frames, output_video_path, fps=30):
    """
    从帧数组保存视频。

    :param frames: 帧数组
    :param output_video_path: 输出视频文件的路径
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame.astype(np.uint8))
    out.release()

def main(args):
    """
    主函数，遍历指定文件夹下的所有视频文件，并将每个视频切分为多个小视频文件。

    :param args: 命令行参数
    """
    input_folder = args.input_folder
    output_folder = args.output_folder
    frame_count = args.frame_count

    # 获取所有视频文件
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

    for video_file in video_files:
        print(f"Processing video file: {video_file}")
        split_video(video_file, output_folder, frame_count=frame_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据帧数将视频切分为小片段。")
    parser.add_argument("input_folder", type=str, help="包含输入视频文件的文件夹路径。")
    parser.add_argument("output_folder", type=str, help="保存输出视频文件的文件夹路径。")
    parser.add_argument("--frame_count", type=int, default=16, help="每个切分视频的帧数")
    args = parser.parse_args()
    main(args)
