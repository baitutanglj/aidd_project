import os
import time
import json
import tempfile
import argparse
import subprocess
from shlex import quote
from rdkit import Chem
from pathlib import Path


def load_parameters(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params

def format_args(args):
    print(f"job_id:{args.job_id}\noutput_dir:{args.output_dir}\nfilename:{args.filename}")
    args_config = argparse.Namespace(**load_parameters(args.config_file))
    args = {**vars(args), **vars(args_config)}
    args = argparse.Namespace(**args)
    args.outfilename = Path(args.filename).stem
    return args


def read_input_file(input_file):
    _, file_extension = os.path.splitext(input_file)
    if file_extension == '':
        mol = Chem.MolFromSmiles(input_file)
    elif file_extension == '.sdf':
        mol = Chem.SDMolSupplier(input_file)[0]
    elif file_extension == '.mol':
        mol = Chem.MolFromMolFile(input_file)
    elif file_extension == '.pdb':
        mol = Chem.MolFromPDBFile(input_file, sanitize=False)
    else:
        mol = None
    return mol


def generate_folder_structure(filepath: str):
    folder_path = os.path.dirname(filepath)
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def gen_temp_file(suffix=None, prefix=None, dir=None, text=True) -> str:
    filehandler, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)
    os.close(filehandler)
    return path

def execute_command(command, arguments, check=True, location=None):
    # arguments = [quote(str(arg)) for arg in arguments]
    complete_command = [command + ' ' + ' '.join(str(e) for e in arguments)]
    old_cwd = os.getcwd()
    if location is not None:
        os.chdir(location)
    # print(complete_command)
    result = subprocess.run(complete_command,
                            check=check,  # force python to raise exception if anything goes wrong
                            universal_newlines=True,  # convert output to string (instead of byte array)
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    os.chdir(old_cwd)
    return result


def wait_until_file_generation(path, interval_sec=1, maximum_sec=10) -> bool:
    counter = 0
    while not os.path.exists(path):
        # wait for an interval
        time.sleep(interval_sec)
        counter = counter + 1

        # if there's time left, proceed
        if maximum_sec is not None and counter * interval_sec >= maximum_sec:
            break
    if os.path.exists(path):
        return True
    else:
        return False



