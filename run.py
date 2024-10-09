import cv2 as cv
import anytrack.app as main
from anytrack.cvgui import GUI
import pandas as pd
import os
import os.path as op
import tkinter as tk
from tkinter import filedialog, ttk
import toml

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
    root = main.App(input_files)
    metadata = {}

    ### batch run manual interactions
    batch = True
    for i, video in enumerate(root.videos):
        metadata[video.name] = {}
        ### background modelling
        bg = root.model_bg(video)
        ### GUI for scale
        if not batch or i == 0:
            gui = GUI(bg, title='Draw a line with 1 cm length.', mode='scale')
            scale = gui.loop()/10. ## TODO: currently length is fixed to 10 mm
        video.set_scale(scale)
        metadata[video.name]['scale'] =  scale
        del gui
        ### GUI for getting odor port positions
        gui = GUI(bg, title='Select odor port positions.', mode='points')
        odor_ports_pos = gui.loop() ## TODO: generalize
        metadata[video.name]['odor_ports'] =  odor_ports_pos

    ### this is the automated processing
    dfs = []
    for video in root.videos:
        ### contours collection
        # TODO:
            # 1. head detection
        cnts = root.collect_contours(video=video, bg=bg)
        ### generate tracks from contours
        # TODO:
            # 1. correct angle
        tracks = root.generate_tracks(cnts)
        ### make DataFrame from tracks
        df = tracks.to_dataframe()
        df['video'] = video.name
        dfs.append(df.loc[:,list(df.columns)[-1:]+list(df.columns)[:-1]])
        ### analyze kinematics
        # TODO:
            # 1. GUI for ROIs, POIs, and estimating scale
            # 2. Transform to physical space
            #   - x,y (y: flipped) SCALED + OFFSET
            #   - major, minor SCALED
            # 3. Calculate forward speed
            # 4. Calculate angular speed
            # 5. Calculate distances
        #print(f'scale = {video.px_per_mm}')
        #df_kine = root.analyze_kinematics(df, scale=video.px_per_mm)

        ### add overlays
        root.video_loop(video, overlay=dict(tracks=tracks))

    outdirname = op.join(root.path, 'anytrack_results')
    os.makedirs(outdirname, exist_ok=True)
    filename = op.join(outdirname, f'tracks.csv')
    pd.concat(dfs).to_csv(filename)
    filename = op.join(outdirname, f'metadata.toml')
    with open(filename, 'w') as f:
        toml.dump(metadata, f)
