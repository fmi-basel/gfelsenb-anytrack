import cv2 as cv
import anytrack.app as main
from anytrack.cvgui import GUI
from anytrack.argsparse import parse_args
import pandas as pd
import os
import os.path as op
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

def savemetadata(data, filepath=None):
    with open(filepath, 'w') as f:
        toml.dump(data, f)

if __name__ == "__main__":
    args = parse_args()

    input_files = open_video_files()
    root = main.App(input_files)
    ### check output
    outdirname = op.join(root.path, 'anytrack_results')
    os.makedirs(outdirname, exist_ok=True)
    metafilename = op.join(outdirname, f'metadata.toml')
    tkroot = tk.Tk()
    tkroot.withdraw()
    metaload = messagebox.askyesno('Metadata file found.', 'Do you want to open existing metadata file?')
    if op.exists(metafilename) and metaload: metadata = toml.load(metafilename)
    else: metadata = {}
    tkroot.update()

    ### batch run manual interactions
    batch = True

    for i, video in enumerate(root.videos):
        ### background modelling
        video.bg = root.model_bg(video)
        if video.name not in metadata:
            metadata[video.name] = {}
            ### GUI for scale
            if not batch or i == 0:
                gui = GUI(video.bg, title='Draw a line with 1 cm length.', mode='scale')
                scale = gui.loop()/10. ## TODO: currently length is fixed to 10 mm
                del gui
            video.set_scale(scale)
            metadata[video.name]['scale'] =  scale
            ### GUI for getting odor port positions
            gui = GUI(video.bg, title='Select odor port positions.', mode='points')
            odor_ports_pos = gui.loop() ## TODO: generalize
            metadata[video.name]['odor_ports'] =  odor_ports_pos
            savemetadata(metadata, filepath=metafilename)
        else:
            scale = metadata[video.name]['scale']
            video.set_scale(scale)

    ### this is the automated processing
    dfs = []
    datafilename = op.join(outdirname, f'tracks.csv')
    flag_processing = True
    if op.exists(datafilename):
        flag_processing = messagebox.askyesno('Data file found.', 'Do you want to overwrite existing data?')
    tkroot.update()
    if flag_processing:
        for video in root.videos:
            ### contours collection
            # TODO:
                # 1. head detection
            cnts = root.collect_contours(video=video, bg=video.bg)
            ### generate tracks from contours
            # TODO:
                # 1. correct angle
            tracks = root.generate_tracks(cnts)
            ### make DataFrame from tracks
            df = tracks.to_dataframe()
            df['video'] = video.name

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
            df_kine = root.analyze_kinematics(df, scale=video.px_per_mm, fps=video.fps)
            flydf = pd.concat([df.loc[:,list(df.columns)[-1:]+list(df.columns)[:-1]], df_kine], axis=1)
            dfs.append(flydf)

            ### add overlays
            if args.show_tracks:
                root.video_loop(video, overlay=dict(tracks=tracks))
        ### save data and metadata
        pd.concat(dfs).to_csv(datafilename)
        savemetadata(metadata, filepath=metafilename)
