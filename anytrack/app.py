import cv2 as cv
from anytrack.background import model_bg
from anytrack.video import Video

class App(object):
    def __init__(self, file):
        self.video = Video(file)
        self.displayname = 'Preview (anytrack v1.0.0)'

    def model_bg(self):
        model_bg(self.video)

    def video_loop(self):
        self.video.loop(self.displayname)
