from collections.abc import Sequence
import os.path as op
import cv2 as cv
import numpy as np
import pandas as pd
import sys

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
        self.overlays = {}

    def add_overlay(self, key, value):
        self.overlays.update({key: value})

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
            ii = self.get_frame_index()
            # if frame is read correctly ret is True
            if frame is None:
                self.reset()
                break
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            ### overlays drawing
            colorimg = frame.copy()
            for k,v in self.overlays.items():
                if k == 'tracks':
                    trace_len = 10 * self.fps
                    x = np.round(v.x[ii]).astype(int)
                    y = np.round(v.y[ii]).astype(int)
                    hx = v.x[ii] +
                    xt = v.x[max(ii-trace_len,0):ii]
                    yt = v.y[max(ii-trace_len,0):ii]
                    trace = np.vstack([xt,yt]).T
                    pts = np.array(trace,np.int32).reshape((-1, 1, 2))
                    ### drawing funcs
                    cv.circle(colorimg, (x,y), 2, (0,255,0), 1)
                    cv.polylines(colorimg, [pts], False, (0,255,0), 1)
            cv.putText( colorimg,
                        f'frame: {self.get_frame_index()}',
                        (10,20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255,0,255),
                        1,
                        cv.LINE_AA,
            )


            cv.imshow(title, colorimg)
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

    def set_scale(self, val):
        self.px_per_mm = val

    def show(self, title, frame=None, waitms=1):
        if frame is None:
            cv.imshow(title, self.frame)
        else:
            cv.imshow(title, frame)
        k = cv.waitKey(waitms) & 0xff # press ESC to exit
        return k
