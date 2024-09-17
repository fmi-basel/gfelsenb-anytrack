import cv2 as cv
import anytrack.app as main
import tkinter as tk
from tkinter import filedialog, ttk

def open_video_files():
    # Hide the root window (no main window)
    root = tk.Tk()
    root.withdraw()
    # Open a file dialog to choose multiple video files
    file_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
    )
    return file_paths


if __name__ == "__main__":
    input_files = open_video_files()
    for input_file in input_files:
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
