import os.path as op
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ---------------------------
# 1) Read 100 equally spaced frames (grayscale)
# ---------------------------
def sample_frames_avi(path, n_samples=100):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        raise ValueError("Could not read frame count from video.")

    idxs = np.linspace(0, n_total - 1, n_samples, dtype=int)

    frames = []
    for idx in tqdm(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"Failed to read frame at index {idx}.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    samples = np.stack(frames, axis=0).astype(np.uint8)  # (N, H, W)
    return samples


# ---------------------------
# 2) Average background frame
# ---------------------------
def average_background(samples_uint8):
    return samples_uint8.mean(axis=0)  # (H, W) float


# ---------------------------
# 3) Per-pixel histogram over observed values (0..255)
#    NOTE: This is memory-heavy for large images: (H*W*256) counts.
# ---------------------------
def per_pixel_histograms(samples_uint8, n_bins=256):
    N, H, W = samples_uint8.shape
    P = H * W
    vals = samples_uint8.reshape(N, P)  # (N, P)

    hists = np.zeros((P, n_bins), dtype=np.uint16)
    for p in range(P):
        hists[p] = np.bincount(vals[:, p], minlength=n_bins)

    return hists.reshape(H, W, n_bins)  # (H, W, 256)


# ---------------------------
# Helper: intersection threshold between two 1D Gaussians in a 2-component GMM
# Returns a threshold between the two means if possible, else midpoint.
# ---------------------------
def gmm_two_gaussian_threshold(weights, means, variances):
    # sort by mean
    order = np.argsort(means)
    w1, w2 = weights[order]
    m1, m2 = means[order]
    v1, v2 = variances[order]
    s1, s2 = np.sqrt(v1), np.sqrt(v2)

    # Solve: w1 * N(x|m1,s1^2) = w2 * N(x|m2,s2^2)
    # Quadratic: a x^2 + b x + c = 0
    a = 1/(2*v1) - 1/(2*v2)
    b = m2/v2 - m1/v1
    c = (m1*m1)/(2*v1) - (m2*m2)/(2*v2) + np.log((w2*s1)/(w1*s2))

    if abs(a) < 1e-12:
        # linear
        if abs(b) < 1e-12:
            return float((m1 + m2) / 2.0)
        x = -c / b
        return float(np.clip(x, 0, 255))

    disc = b*b - 4*a*c
    if disc < 0:
        return float((m1 + m2) / 2.0)

    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)

    lo, hi = min(m1, m2), max(m1, m2)
    candidates = [r for r in (r1, r2) if lo <= r <= hi]
    if candidates:
        x = candidates[0] if len(candidates) == 1 else min(candidates, key=lambda r: abs(r - (m1+m2)/2))
        return float(np.clip(x, 0, 255))

    return float((m1 + m2) / 2.0)


# ---------------------------
# 4) Fit per-pixel GMM (1 or 2 comps), pick best by BIC.
#    Background model = mean (1-comp) OR brighter mean (2-comp).
#    Also returns per-pixel threshold if 2-comp chosen (else NaN).
# ---------------------------
def fit_gmm_background(samples_uint8, bic_improvement=10.0, lowp=120., min_std=3.0, reg_covar=1e-3, random_state=0):
    N, H, W = samples_uint8.shape
    P = H * W
    vals = samples_uint8.reshape(N, P).astype(np.float32)  # (N, P)

    bg = np.zeros(P, dtype=np.float32)
    thr = np.full(P, np.nan, dtype=np.float32)

    for p in tqdm(range(P)):
        X = vals[:, p].reshape(-1, 1)

        if np.unique(X).size < 5 or X.std() < min_std:
            bg[p] = float(X.mean())
            thr[p] = np.nan
            continue

        g1 = GaussianMixture(n_components=1, covariance_type="full",
                             reg_covar=reg_covar, random_state=random_state)
        g1.fit(X)
        bic1 = g1.bic(X)

        g2 = GaussianMixture(n_components=2, covariance_type="full",
                             reg_covar=reg_covar, random_state=random_state)
        g2.fit(X)
        bic2 = g2.bic(X)

        use_two = (bic2 + bic_improvement) < bic1  # require a small improvement to avoid overfitting

        if not use_two:
            bg[p] = float(g1.means_[0, 0])
        else:
            weights = g2.weights_
            means = g2.means_.flatten()
            variances = np.array([g2.covariances_[k, 0, 0] for k in range(2)], dtype=np.float32)

            # background = brighter component mean
            bg[p] = float(np.max(means))
            if bg[p] < lowp:
                bg[p] = lowp
            thr[p] = gmm_two_gaussian_threshold(weights, means, variances)

    bg_img = bg.reshape(H, W)
    thr_img = thr.reshape(H, W)
    return bg_img, thr_img

def detect_four_arenas(bg_u8, min_area_frac=0.01):
    """
    bg_u8: HxW uint8 grayscale background image
    returns: circles, arena_mask
      circles = [(cx, cy, r), ...] length 4 (sorted by area desc)
      arena_mask = HxW uint8 {0,255} mask of all arenas combined
    """
    H, W = bg_u8.shape[:2]

    # 1) Smooth + threshold (bright arenas on dark background)
    blur = bg_u8 #cv2.GaussianBlur(bg_u8, (0, 0), 3)
    ret, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #print("Calculated Otsu threshold value:", ret)
    #ret, bw = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)


    # 2) Morph cleanup (remove specks, fill small gaps)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k, iterations=1)
    #bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    # 3) Connected components
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # stats: [label, x, y, w, h, area] but with OpenCV it's [x,y,w,h,area] per label
    # label 0 is background
    comps = []
    min_area = int(min_area_frac * H * W)
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        comps.append((area, mask))

    if len(comps) < 4:
        raise RuntimeError(f"Found only {len(comps)} candidate arenas. Try lowering min_area_frac.")

    # 4) Keep 4 largest components
    comps.sort(key=lambda x: x[0], reverse=True)
    comps = comps[:4]

    circles = []
    arena_mask = np.zeros((H, W), dtype=np.uint8)

    for area, comp_mask in comps:
        arena_mask = cv2.bitwise_or(arena_mask, comp_mask)

        # Fit circle using contour points
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circles.append((float(cx), float(cy), float(r)))

    return circles, arena_mask

def gaussian_blur_on_mask(bg_u8, arena_mask, ksize=(0, 0), sigma=10):
    """
    bg_u8:       HxW uint8 grayscale
    arena_mask:  HxW uint8 (0/255) or bool mask of arena pixels
    ksize:       (0,0) lets OpenCV pick from sigma; or e.g. (31,31)
    sigma:       Gaussian sigma
    """
    mask = arena_mask.astype(bool)

    blurred = cv2.GaussianBlur(bg_u8, ksize, sigmaX=sigma)

    out = bg_u8.copy()
    out[mask] = blurred[mask]
    return out

def signed_change_overlay(gray_u8, bg_u8, thr=10, scale=8, arena_mask=None):
    """
    Returns an HxWx3 uint8 image on white background:
      brighter-than-bg -> green tint
      darker-than-bg   -> magenta tint
    thr:   minimum absolute change to visualize
    scale: how quickly saturation increases with |diff|
    arena_mask: optional HxW mask (0/255 or bool) to restrict visualization
    """
    gray = gray_u8.astype(np.int16)
    bg   = bg_u8.astype(np.int16)
    diff = gray - bg  # signed

    if arena_mask is not None:
        m = arena_mask.astype(bool)
    else:
        m = np.ones(gray_u8.shape, dtype=bool)

    bright = (diff >  thr) & m
    dark   = (diff < -thr) & m

    sat = np.clip(np.abs(diff) * scale, 0, 255).astype(np.uint8)

    out = np.full((*gray_u8.shape, 3), 255, dtype=np.uint8)  # white (BGR)

    # Brighter: tint green by reducing R&B
    out[bright, 0] = 255 - sat[bright]   # B
    out[bright, 1] = 255                 # G
    out[bright, 2] = 255 - sat[bright]   # R

    # Darker: tint magenta by reducing G
    out[dark,   0] = 255                 # B
    out[dark,   1] = 255 - sat[dark]     # G
    out[dark,   2] = 255                 # R

    return out

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    avi_folder = "/Users/golddenn/Desktop/test_sample"
    avi_path = op.join(avi_folder, "localsearch_2025-12-11T09_34_21.avi")
    bg_file = op.join(avi_folder, "background_gmm.png")

    if op.isfile(bg_file):
        gmm_bg_u8 = cv2.imread(bg_file, cv2.IMREAD_GRAYSCALE)
        circles, arena_mask = detect_four_arenas(gmm_bg_u8, min_area_frac=0.01)
        # Visualize detections
        vis = cv2.cvtColor(gmm_bg_u8, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r) in circles:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 2)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

        cv2.imwrite(op.join(avi_folder, "arenas_mask.png"), arena_mask)
        cv2.imwrite(op.join(avi_folder, "arenas_detected.png"), vis)
        print("Circles:", circles)
        bg_blurred = gaussian_blur_on_mask(gmm_bg_u8, arena_mask, sigma=3.)
        cv2.imwrite(op.join(avi_folder, "background_blur_in_arenas.png"), bg_blurred)

        cap = cv2.VideoCapture(avi_path)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            overlay = signed_change_overlay(gray, bg_blurred, thr=20, scale=15, arena_mask=arena_mask)

            cv2.imshow("Signed changes (green=brighter, magenta=darker)", overlay)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        samples = sample_frames_avi(avi_path, n_samples=100)
        #x0,y0=1167,1741
        #s=52
        #samples = samples[:,y0:y0+s,x0:x0+s]
        avg_bg = average_background(samples)  # float background from simple averaging
        # Optional: per-pixel histograms (can be huge)
        # hists = per_pixel_histograms(samples)

        gmm_bg, gmm_thresholds = fit_gmm_background(samples, min_std=10.0, bic_improvement=0.0)

        # Convert to uint8 images if you want to save/view
        avg_bg_u8 = np.clip(avg_bg, 0, 255).astype(np.uint8)
        gmm_bg_u8 = np.clip(gmm_bg, 0, 255).astype(np.uint8)


        cv2.imwrite(op.join(avi_folder, "background_avg.png"), avg_bg_u8)
        cv2.imwrite(op.join(avi_folder, "background_gmm.png"), gmm_bg_u8)
        # thresholds are per-pixel (NaN where 1-Gaussian selected)
        np.save(op.join(avi_folder, "thresholds.npy"), gmm_thresholds)