import cv2 as cv
import numpy as np
from tqdm.auto import tqdm

from anytrack.video import  CentroidTracks, ContoursCollection, Video

def get_contours(bg, frame, mask=None, threshold_tracking=30):
    """
    Get contours from background subtraction

    Returns contours

    Parameters:
        - bg: background image
        - frame: current frame
        - threshold_tracking (opt, default: 30): threshold for contour finding
    """
    diff = cv.subtract(bg, frame)
    __, subtr = cv.threshold(cv.cvtColor(diff, cv.COLOR_BGR2GRAY), threshold_tracking, 255, cv.THRESH_BINARY)
    if mask is None: masked = subtr
    else: masked = cv.bitwise_and(subtr.astype(np.uint8), mask[:,:,0])
    contours, hierarchy = cv.findContours(masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return contours, diff, subtr

class App(object):
    def __init__(self, file):
        self.video = Video(file)
        self.displayname = 'Preview (anytrack v1.0.0)'

    def collect_contours(self, video=None, bg=None, verbose=True, onlynframes=None, display=False):
        if video is None: video = self.video
        if bg is None: bg = self.bg
        if onlynframes is not None: video.nframes = onlynframes 
        cnts = ContoursCollection() ### container for all contours and centroids
        ### iterate through video
        for i in tqdm(range(video.nframes), desc='Tracking on frame:', disable=(not verbose)):
            frame = video.read()
            if frame is None: continue
            colorimg = frame.copy()
            ### bg subtraction
            mask = np.zeros(frame.shape, dtype=np.uint8)
            #cv.circle(mask, roi[:2], roi[2], 255, -1)
            contours, diff, subtr = get_contours(bg, frame, threshold_tracking=40)
            contours = [el for el in contours if len(el) > 5]
            contours = sorted(contours, key=cv.contourArea, reverse=True) ### largest first
            ### draw on top
            cv.drawContours(colorimg, contours, 0, (255.,0.,255.), 1)
            ### get centroids
            centroids = [cv.fitEllipse(cnt) for cnt in contours]
            if len(centroids) > 0:
                e = centroids[0]
                cv.ellipse(colorimg, e, (0,255.,0.))
                pos = np.array(e[0]).astype(np.uint32)
                cv.circle(colorimg, (pos[0], pos[1]), 2, (0,255.,0.), 1)
            ### display video
            if display:
                title = f'{video.name}'
                k = video.show(title, frame=colorimg) ## waitms = milliseconds of waiting (0 = forever)
                if k == 27 or cv.getWindowProperty(title, 0)<0: break
            ### adding to collection
            cnts.add(video.get_frame_index(), contours, centroids)
        return cnts

    def generate_tracks(self, cnts):
        """
        Current implementation only takes largest contour into account.
        """
        tracks = CentroidTracks(len(cnts))
        for i, coid in enumerate(cnts):
            tracks.frame[i] = cnts.frameindex[i]
            if len(coid) > 0: ### only if contour exists
                cc = coid[0] ### largest contour
                tracks.x[i] = cc[0][0]
                tracks.y[i] = cc[0][1]
                tracks.major[i] = cc[1][1]
                tracks.minor[i] = cc[1][0]
                tracks.angle[i] = cc[2]
        return tracks

    def model_bg(self, video, bgframes=90, niters=0, ghost_thr=[10, 30], verbose=True, display=False):
        """
        Background Modelling using Iterative Average and Ghost Subtraction

        Returns background image as unsigned 8-bit integer ndarray

        Parameters:
            - video: video object
            - bgframes (opt, default: 90): number of frames for averaging background
            - niters (opt, default: 0): number of iterations for removing ghost artifacts
        """
        ### random sampling of frames from video
        choices = np.random.choice(video.nframes, size=bgframes)
        frames = []
        ### first averaging
        for i in tqdm(choices, desc='Averaging frame for background:', disable=(not verbose)):
            video.set_frame(i)
            frames.append(video.read())
        avg_img = np.mean(frames, axis=0)
        avg_img = avg_img.astype(np.uint8)
        if display: video.show('Background model', frame=avg_img, waitms=0)

        ### iterative ghost subtraction
        for j in range(niters):
            newbg = np.zeros(frames[0].shape, dtype=np.float64)
            bgcount = np.zeros(frames[0].shape, dtype=np.float64)
            bgcount[:] = 1.
            choices = np.random.choice(video.nframes, size=int(bgframes))
            for i in tqdm(choices, desc=f'Iteration {j+1}:'):
                video.set_frame(i)
                frame = video.read()
                difference1 = cv.subtract(avg_img, frame)[:,:,0]
                __, subtr1 = cv.threshold(difference1, ghost_thr[0], 255, cv.THRESH_BINARY)
                difference2 = cv.subtract(frame, avg_img)[:,:,0]
                __, subtr2 = cv.threshold(difference2, ghost_thr[1], 255, cv.THRESH_BINARY)
                subtr = cv.bitwise_or(subtr1, subtr2)

                ##subtr = subtr1
                bgmask = np.zeros(frames[0].shape, dtype=np.uint8)
                bgmask[subtr==0] = frame[subtr==0]
                bgcount[subtr==0] += 1.
                newbg += bgmask.astype(np.float64)

                avg = np.clip(np.divide(newbg,bgcount), 0, 255).astype(np.uint8)
                video.show('Iterative average', frame=avg)
            avg_img[:,:,0] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
            avg_img[:,:,1] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
            avg_img[:,:,2] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
            avg_img = avg_img.astype(np.uint8)
        bg = avg_img.astype(np.uint8)
        video.reset()
        video.close_all()
        return bg

    def video_loop(self):
        self.video.loop(self.displayname)
