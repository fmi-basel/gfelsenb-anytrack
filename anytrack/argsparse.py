import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run antrack."
    )
    #parser.add_argument(
    #    "--config",
    #    help="Path to config TOML file. If omitted, uses the default from config."
    #)

    ### verbose option
    parser.add_argument('-st', '--show_tracks',
                        action='store_true')

    args = parser.parse_args()
    return args
