import cv2 as cv
import anytrack as track
import anytrack.app as main


if __name__ == "__main__":
    input_file = "/Users/golddenn/Desktop/temp-08062024162734.avi"
    root = main.App(input_file)
    video = root.video
    ### background modelling
    bg = root.model_bg(video)
    ### contours collection
    cnts = root.collect_contours(bg)
    ### generate tracks from contours
    track = root.generate_tracks(cnts)
    ### add overlays

    #root.video_loop()
