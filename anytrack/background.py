import cv2 as cv
import numpy as np

def model_bg(video, bgframes=100, niters=2, ghost_thr=[30, 30]):
    """
    Background Modelling using Iterative Average and Ghost Subtraction

    Returns background image as unsigned 8-bit integer ndarray

    Parameters:
        - video: video object
        - bgframes (opt, default: 100): number of frames for averaging background
        - niters (opt, default: 2): number of iterations for removing ghost artifacts
    """
    ### random sampling of frames from video
    choices = np.random.choice(video.nframes, size=bgframes)
    frames = []
    ### first averaging
    for i in tqdm(choices, desc='First average:'):
        video.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        frames.append(video.cap.read()[1])
    avg_img = np.mean(frames, axis=0)
    avg_img = avg_img.astype(np.uint8)

    bgfolder = op.join(video.outf, 'bg', f'{video.name}')
    os.makedirs(bgfolder, exist_ok=True)
    bgfile = op.join(bgfolder, f'{video.name}_0.png')
    cv2.imwrite(bgfile, avg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    ### iterative ghost subtraction
    for j in range(niters):
        newbg = np.zeros(frames[0].shape, dtype=np.float64)
        bgcount = np.zeros(frames[0].shape, dtype=np.float64)
        bgcount[:] = 1.
        choices = np.random.choice(video.nframes, size=int(bgframes))
        for i in tqdm(choices, desc=f'Iteration {j+1}:'):
            video.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame = video.cap.read()[1]
            difference1 = cv2.subtract(avg_img, frame)[:,:,0]
            __, subtr1 = cv2.threshold(difference1, ghost_thr[0], 255, cv2.THRESH_BINARY)
            difference2 = cv2.subtract(frame, avg_img)[:,:,0]
            __, subtr2 = cv2.threshold(difference2, ghost_thr[1], 255, cv2.THRESH_BINARY)
            subtr = cv2.bitwise_or(subtr1, subtr2)

            ##subtr = subtr1
            bgmask = np.zeros(frames[0].shape, dtype=np.uint8)
            bgmask[subtr==0] = frame[subtr==0]
            bgcount[subtr==0] += 1.
            newbg += bgmask.astype(np.float64)

            avg = np.clip(np.divide(newbg,bgcount), 0, 255).astype(np.uint8)
            #cv2.imshow("avg", cv2.resize(avg, (700,700)))
            #k = cv2.waitKey(1) & 0xff # press ESC to exit
            #if k == 27 or cv2.getWindowProperty('avg', 0)<0: break
        avg_img[:,:,0] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
        avg_img[:,:,1] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
        avg_img[:,:,2] = np.clip(np.divide(newbg[:,:,0],bgcount[:,:,0]), 0, 255)
        avg_img = avg_img.astype(np.uint8)

        bgfolder = op.join(video.outf, 'bg', f'{video.name}')
        bgfile = op.join(bgfolder, f'{video.name}_{j+1}.png')
        cv2.imwrite(bgfile, avg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    bg = avg_img.astype(np.uint8)
    bgfolder = op.join(video.outf, 'bg')
    bgfile = op.join(bgfolder, f'{video.name}_bg.png')
    cv2.imwrite(bgfile, avg_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return bg
