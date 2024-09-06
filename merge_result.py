import os
import math
import pickle
from rdkit import Chem
from pathlib import Path
from glob import glob
import pandas as pd
from utils import execute_command, wait_until_file_generation

def merge_output(output_dir, topk, task):
    outfilename = ""
    result_flag = False
    if not isinstance(topk, int):
        topk_result = execute_command("tail", ["-n +2 -q", output_dir+"/*.csv", "| wc -l"])
        topk = math.floor(int(topk_result.stdout.splitlines()[0])*topk)
    task = Path(task).stem

    if task == "morgan_similarity":
        if len(glob(output_dir+"/*.csv"))>0:
            outfilename = os.path.join(output_dir, "out_morgan_similarity.csv")
            command = 'csvstack'
            arguments = [output_dir+"/*.csv", "| csvsort -c similarity -r | head -n", topk+1, ">", outfilename]
            result = execute_command(command, arguments)
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            _ = execute_command('find',
                                [output_dir, "-type", "f", "-name", "*.csv", "! -name", os.path.basename(outfilename),
                                 "-delete"])
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            if result_flag:
                print(f"Job output merged successfully: {outfilename}")
        else:
            result_flag = False
            print("No job output found.")


    elif task == "sub_match":
        if len(glob(output_dir + "/*.csv")) > 0:
            outfilename = os.path.join(output_dir, "out_sub_match.csv")
            _ = execute_command('head', ['-n 1 ', glob(output_dir + "/*.csv")[0], '>', outfilename])
            command = 'tail -n +2 -q'
            arguments = [output_dir + "/*.csv", " | shuf -n ", topk, ">>", outfilename]
            result = execute_command(command, arguments)
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            _ = execute_command('find',
                                [output_dir, "-type", "f", "-name", "*.csv", "! -name", os.path.basename(outfilename),
                                 "-delete"])

            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            if result_flag:
                print(f"Job output merged successfully: {outfilename}")
        else:
            result_flag = False
            print("No job output found.")

    elif task == "shape_it_similarity":
        if len(glob(output_dir+"/*.csv"))>0 and len(glob(output_dir+"/*.sdf"))>0:
            outfilename = os.path.join(output_dir, "out_3D_similarity_shape_it.csv")
            command = 'csvstack'
            arguments = [output_dir+"/*.csv", "| csvsort -c 'Shape-it::Tanimoto' -r | head -n", topk+1, ">", outfilename]
            result = execute_command(command, arguments)
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            _ = execute_command('find',
                                [output_dir, "-type", "f", "-name", "*.csv", "! -name", os.path.basename(outfilename),
                                 "-delete"])
            _ = execute_command('find',
                                [output_dir, "-type", "f", "-name", "*.sdf", "-delete"])
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            if result_flag:
                print(f"Job output merged successfully: {outfilename}")
        else:
            result_flag = False
            print("No job output found.")

    elif task == "vina_docking":
        if len(glob(output_dir+"/*.csv"))>0:
            outfilename = os.path.join(output_dir, "out_vina_docking.csv")
            command = 'csvstack'
            arguments = [output_dir + "/*.csv", "| csvsort -c score | head -n", topk + 1, ">",
                         outfilename]
            result = execute_command(command, arguments)
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            _ = execute_command('find',
                                [output_dir, "-type", "f", "-name", "*.csv", "! -name", os.path.basename(outfilename),
                                 "-delete"])
            result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
            if result_flag:
                print(f"Job output merged successfully: {outfilename}")
        else:
            result_flag = False
            print("No job output found.")

    # elif task == "vina_docking":
    #     if len(glob(output_dir+"/*.pkl"))>0:
    #         outfilename = os.path.join(output_dir, "out_vina_docking.sdf")
    #         fnames = glob(output_dir+"/*.pkl")
    #         results_df = pd.DataFrame()
    #         for fname in fnames:
    #             with open(fname, 'rb') as f:
    #                 df_tmp = pickle.load(f)
    #                 results_df = pd.concat([results_df, df_tmp], ignore_index=True)
    #         results_df.sort_values('score', ascending=True, inplace=True, ignore_index=True)
    #         results_df = results_df[:topk]
    #         with Chem.SDWriter(outfilename) as f:
    #             for idx, row in results_df.iterrows():
    #                 mol = row['mol']
    #                 mol.SetProp("_Name", row['conformer_name'])
    #                 mol.SetProp('SCORE', str(row['score']))
    #                 f.write(mol)
    #         _ = execute_command('find',
    #                             [output_dir, "-type", "f", "-name", "*.pkl", "-delete"])
    #         result_flag = wait_until_file_generation(outfilename, interval_sec=1, maximum_sec=10)
    #     else:
    #         result_flag = False
    #         print("No job output found.")

    if result_flag:
        print("merge job output completed successfully. Output file: ", outfilename)
    return result_flag