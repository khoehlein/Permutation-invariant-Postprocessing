import os
import argparse
import zipfile
from shutil import copyfile
from multiprocessing import Pool


def handle_file(input_tuple):
    source_path, target_path, fi = input_tuple
    print(f'[INFO] Processing file {source_path}/{fi}')
    if fi.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(source_path, fi), 'r') as zip_ref:
            zip_ref.extractall(target_path)
    elif fi.endswith('.nc'):
        copyfile(os.path.join(source_path, fi), os.path.join(target_path, fi))
    else:
        raise Exception(f'[ERROR] Unknown source file type: {fi}')


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='source directory')
    parser.add_argument('target', type=str, help='target directory')
    args = parser.parse_args()
    folders = os.listdir(args.source)
    for fo in folders:
        source_path = os.path.join(args.source, fo)
        target_path = os.path.join(args.target, fo)
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
        files = os.listdir(source_path)
        inputs = [(source_path, target_path, fi) for fi in files]
        with Pool(8) as p:
            p.map(handle_file, inputs)

    folders = os.listdir(args.target)
    for fo in folders:
        target_path = os.path.join(args.target, fo)
        files = os.listdir(target_path)
        for fi in files:
            assert fi.endswith('.nc'), f'[ERROR] Unknown target file type: {fi}'
