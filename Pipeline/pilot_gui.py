import argparse
import os

def parse_config_file(config_file):
    config_file=open(config_file,"r")
    print(config_file)
    lines=config_file.readlines()
    print(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pilot gui according taking in as input the config file.')
    parser.add_argument('--config_path', help='Path to the config file')
    args = parser.parse_args()
    config_file=args.config_path
    parse_config_file(os.path.normpath(str(config_file)))
