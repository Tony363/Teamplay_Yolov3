import argparse
import moviepy.editor as mpe
#make audio
def addAudio(videoPath, audioVideoPath,speed:int):

    videoClip = mpe.VideoFileClip(videoPath)
    audioVideoClip = mpe.VideoFileClip(audioVideoPath)

    final_clip = videoClip.set_audio(audioVideoClip.audio)
    final_clip = final_clip.speedx(factor=speed)
    
    videoPath = videoPath[:-4] + "_audio.mp4"
    final_clip.write_videofile(videoPath)



def readCommand():
	# construct the argument parse and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoPath", help="path to the stitched video file")
    parser.add_argument("--audioVideoPath", help="path to one of the original video to retrieve its audio channel")

    args = vars(parser.parse_args())
    return args


if __name__=="__main__":
    args = readCommand()
    addAudio(args['videoPath'], args['audioVideoPath'])