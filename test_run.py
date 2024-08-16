import cv2 as cv
import anytrack.app as main


if __name__ == "__main__":
    input_file = "C:/Users/golddenn/Desktop/FlyQuarter/temp-08062024170708_rightACV.avi" #"/Users/golddenn/Desktop/temp-08062024170708_rightACV.avi"  #"/Users/golddenn/Desktop/temp-08062024162734.avi"
    root = main.App(input_file)
    video = root.video
    ### background modelling
    bg = root.model_bg(video)
    ### contours collection
    cnts = root.collect_contours(video=video, bg=bg)
    ### generate tracks from contours
    tracks = root.generate_tracks(cnts)
    ### make DataFrame from tracks
    df = tracks.to_dataframe()
    ### add overlays
    root.video_loop(overlay=dict(tracks=tracks))
