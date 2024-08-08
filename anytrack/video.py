import os.path as op
import cv2 as cv

class CentroidTracks(object):
    def __init__(self, nrows, ncols=5, colnames=['x', 'y']):
        """
        nrows: number of rows, i.e. timesteps
        ncols: number of columns (default = 5 for OpenCV ellipse fitting)
        """
        pass

    def __add__(self, other):
        return self

class ContoursCollection(object):
    def __init__(self):
        pass

class Video(object):
    def __init__(self, file):
        self.name = op.basename(file).split('.')[0]
        self.dir = op.dirname(file)
        self.fullpath = file
        self.cap = cv.VideoCapture(file)
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.nframes = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))
        self.frame = None

    def close(self, title):
        self.cap.release()
        cv.destroyWindow(title)

    def loop(self, title):
        cap = self.cap ## local ref
        while cap.isOpened():
            frame = self.read()

            # if frame is read correctly ret is True
            if frame is None:
                self.reset()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            cv.imshow(title, gray)
            k = cv.waitKey(1)
            if k == 27:
                break
            elif k == -1:
                pass
            else:
                print(k)
        self.close(title)

    def read(self):
        ret, frame = self.cap.read()
        self.frame = frame
        if ret:
            return frame
        else:
            return None

    def reset(self):
        self.cap.release()
        self.cap = cv.VideoCapture(self.fullpath)

    def set_frame(self, i):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, i)

    def show(self, title, frame=None, waitms=1):
        if frame is None:
            cv.imshow(title, self.frame)
        else:
            cv.imshow(title, frame)
        k = cv.waitKey(waitms) & 0xff # press ESC to exit
        return k
