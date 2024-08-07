import cv2 as cv
import anytrack as track
import anytrack.app as main


if __name__ == "__main__":
    input_file = "/Users/golddenn/Desktop/temp-08062024162734.avi"
    root = main.App(input_file)
    ### background modelling
    bg = root.model_bg()

    #root.video_loop()
