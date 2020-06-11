import argparse
import moviepy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cutVideo(videoPath,startTime,endTime, subVideoName = "result.mp4"):
    ffmpeg_extract_subclip(videoPath, int(startTime), int(endTime), targetname=subVideoName)

def readCommand():
    parser = argparse.ArgumentParser()
    parser.add_argument("videoPath", help="path of the video file")
    parser.add_argument("startTime", help="start time in seconds")
    parser.add_argument("endTime", help="end time in seconds")
    parser.add_argument("-o", "--output", help="output video name")
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = readCommand()
    if args["output"] is None:
        cutVideo(args["videoPath"], args["startTime"], args["endTime"])
    else:
        cutVideo(args["videoPath"], args["startTime"], args["endTime"], args["output"])