import os
import argparse
from pathlib import Path
from tqdm import tqdm

from config import PathsConfig, TypesConfig
from model import VAD


def main(model_path: str, output_path: str):
    '''
    Infers the trained model saved in model_path on the all files from PathsConfig.infer folder.
    Saves results in readable format to output_path.
    Parameters:
        model_path (str): path to the trained model to use
        output_path (str): path to the output text file
    '''
    vad = VAD(model_path)
    result = ''
    for dirpath, dirnames, filenames in os.walk(PathsConfig.infer):
        for filename in filenames:
            if filename.endswith(f".{TypesConfig.infer}"):
                path = Path(dirpath) / filename
                output = vad(path)
                print(filename)
                result += filename + ', ' + str(output.tolist())
    with open(output_path, 'w') as f:
        f.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to a trained model')
    parser.add_argument('-o', '--output', type=str, default=Path(PathsConfig.meta)/'pred.txt',
                        help='Path to an output txt file')
    args = parser.parse_args()
    main(args.model, args.output)