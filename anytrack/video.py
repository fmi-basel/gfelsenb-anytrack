from collections.abc import Sequence
import os.path as op
import cv2 as cv
import numpy as np
import pandas as pd

class CentroidTracks():
    def __init__(self, nrows, colnames=['frame', 'x', 'y', 'angle', 'major', 'minor']):
        """
        nrows: number of rows, i.e. timesteps
        colnames: column names (default = 1 frame index + 5 for OpenCV ellipse fitting)
        """
        self.ncols = len(colnames)
        self.columns = colnames
        for col in colnames:
            setattr(self, col, np.empty(nrows) * np.nan)

    def to_dataframe(self):
        data = np.stack([self.__dict__[col] for col in self.columns], axis=1)
        df = pd.DataFrame(data=data, columns=self.columns)
        # force frame index to be integer
        df['frame'] = df['frame'].astype(int) 
        return df

class ContoursCollection(Sequence):
    def __init__(self):
        self.frameindex = []
        self.contours = []
        self.centroids = []

    def __getitem__(self, index):
        return self.centroids[index]

    def __len__(self):
        return len(self.contours)

    def add(self, i, cnt, coid):
        self.frameindex.append(i)
        self.contours.append(cnt)
        self.centroids.append(coid)

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
    
    def close_all(self):
        cv.destroyAllWindows()

    def get_frame_index(self):
        return int(self.cap.get(cv.CAP_PROP_POS_FRAMES))

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
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    def set_frame(self, i):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, i)

    def show(self, title, frame=None, waitms=1):
        if frame is None:
            cv.imshow(title, self.frame)
        else:
            cv.imshow(title, frame)
        k = cv.waitKey(waitms) & 0xff # press ESC to exit
        return k
