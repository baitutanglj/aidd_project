import os
import re
import time
import json
import math
import pickle
import shutil
import argparse
import tempfile
import subprocess
import pandas as pd
from glob import glob
from shlex import quote
from time import time
from rdkit import Chem
from pathlib import Path
from functools import partial
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit.Chem import AllChem, DataStructs
from multiprocessing import Pool, cpu_count
from utils import *

def read_molecule_names_from_pdbqt(molecule_file):
    with open(molecule_file, 'r')as f:
        lines = f.read()
    pattern = r'REMARK\s+Name\s+=\s+(\S+)'
    names = re.findall(pattern, lines)
    return names



def fix_pdb(receptor_pdb_path):
    print("------fix receptor pdb------")
    temp_pdb_file = gen_temp_file(suffix=".pdb")
    fixer = PDBFixer(filename=receptor_pdb_path)
    # fix_pdb: perform fixes specified in the configuration
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    print(f"Replaced {len(fixer.nonstandardResidues)} non-standard residues.")
    fixer.removeHeterogens(keepWater=True)  # "Removed heterogens."
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print(f"Added {len(fixer.missingAtoms)} missing atoms.")
    fixer.addMissingHydrogens(pH=7.0)  # "Added missing hydrogens."
    PDBFile.writeFile(fixer.topology, fixer.positions, open(temp_pdb_file, 'w'))
    print(f"Wrote fixed PDB to file {temp_pdb_file}.")

    return temp_pdb_file


def export_as_pdb2pdbqt(target_pdb):
    # generate temporary copy
    target = Chem.MolFromPDBFile(target_pdb, sanitize=False)
    temp_target_pdb = gen_temp_file(suffix=".pdb")
    temp_target_pdbqt = gen_temp_file(suffix=".pdbqt")
    Chem.MolToPDBFile(mol=target, filename=temp_target_pdb)

    arguments = [temp_target_pdb, '-opdbqt', '-O', temp_target_pdbqt, '-xr', '-p', 7.4, '--partialcharge', 'gasteiger']
    result = execute_command('obabel', arguments, check=False, location=None)

    # clean up the temporary file
    if os.path.exists(temp_target_pdb):
        os.remove(temp_target_pdb)
        os.remove(target_pdb)
    print(f"Exported target as PDBQT file {temp_target_pdbqt}.")
    return temp_target_pdbqt


def export_as_pdbqt2sdf(pdbqt_docked):
    path_sdf_result = gen_temp_file(suffix='.sdf', dir='/tmp')
    obabel_arguments = [pdbqt_docked, '-ipdbqt', '-osdf', '-O', path_sdf_result]
    obabel_result = execute_command('obabel', obabel_arguments, check=False)
    obabel_flag = wait_until_file_generation(path_sdf_result)

    return path_sdf_result


def extract_box(reference_ligand_path):
    ref_mol = read_input_file(reference_ligand_path)
    box = None
    if ref_mol is not None:
        # extract coordinates
        x_coords = [atom[0] for atom in ref_mol.GetConformer(0).GetPositions()]
        y_coords = [atom[1] for atom in ref_mol.GetConformer(0).GetPositions()]
        z_coords = [atom[2] for atom in ref_mol.GetConformer(0).GetPositions()]
        if x_coords is not None:
            def dig(value):
                return round(value, ndigits=2)
            size_x = dig(max(x_coords) - min(x_coords))
            size_y = dig(max(y_coords) - min(y_coords))
            size_z = dig(max(z_coords) - min(z_coords))
            center_x = dig(sum(x_coords)/len(x_coords))
            center_y = dig(sum(y_coords)/len(y_coords))
            center_z = dig(sum(z_coords)/len(z_coords))
            print(f"X coordinates: min={dig(min(x_coords))}, max={dig(max(x_coords))}, mean={center_x}")
            print(f"Y coordinates: min={dig(min(y_coords))}, max={dig(max(y_coords))}, mean={center_y}")
            print(f"Z coordinates: min={dig(min(z_coords))}, max={dig(max(z_coords))}, mean={center_z}")
            box = {'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
                   'center_x': center_x, 'center_y': center_y, 'center_z': center_z}

    return box

    
def target_preparator(receptor_pdb_path, reference_ligand_path):
    temp_pdb_file = fix_pdb(receptor_pdb_path)
    #AutodockVinaTargetPreparator
    temp_target_pdbqt = export_as_pdb2pdbqt(target_pdb=temp_pdb_file)

    ## if there is a reference ligand provided, calculate mean, minimum and maximum coordinates and log out
    box = extract_box(reference_ligand_path)
    # if box is not None:
    #     size_x, size_y, size_z, center_x, center_y, center_z = box
    #     print(f"Size of the box: X={size_x}, Y={size_y}, Z={size_z}")
    #     print(f"Center of the box: X={center_x}, Y={center_y}, Z={center_z}")
    
    return temp_target_pdbqt, box
    
def extract_score_from_VinaResult(molecule) -> str:
    result_tag_lines = molecule.GetProp('REMARK').split("\n")
    result_line = [line for line in result_tag_lines if 'VINA RESULT' in line][0]
    parts = result_line.split()
    return float(parts[2])

def integrate_result(path_sdf_result, mol_name, ligand_name):
    if os.path.getsize(path_sdf_result) == 0:
        return None
    mols = Chem.SDMolSupplier(path_sdf_result, removeHs=False)
    idx = 0
    best = True
    result = []
    for mol in mols:
        try:
            Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            print(ligand_name, 'Could not read molecule')
            break
        # extract the score from the AutoDock Vina output and update some tags
        score = extract_score_from_VinaResult(molecule=mol)
        name = f"{ligand_name}_{mol_name}_{idx}"
        # mol.SetProp("_Name", name)
        # mol.SetProp('SCORE', score)
        mol.ClearProp('REMARK')
        result.append([name, mol_name, idx, score, Chem.MolToSmiles(mol), best, mol])
        idx += 1
        best = False
    os.remove(path_sdf_result)
    if len(result)==1:
        return result[0]
    return result

def extract_affinity(text):
    # 使用正则表达式匹配 affinity 值
    match = re.search(r'^\s*1\s+(-?\d+\.\d+)', text, re.MULTILINE)
    if match:
        return float(match.group(1))
    else:
        return None


def get_docking_result(docking_result, topk):
    results_df = pd.DataFrame(docking_result, columns=['ligand_original', 'ligand_docked', 'score', 'ligand_name', 'mol_name'])
    results_df.sort_values('score', ascending=True, inplace=True, ignore_index=True)
    topk = topk if isinstance(topk, int) else math.floor(len(results_df)*topk)
    results_df = results_df[:topk]

    return results_df



def docking_job(ligand, args):
    mol_name = read_molecule_names_from_pdbqt(ligand)[0]
    ligand_name = Path(ligand).stem
    cur_tmp_output_dir = tempfile.mkdtemp()
    tmp_pdbqt_docked = gen_temp_file(suffix='.pdbqt', dir=cur_tmp_output_dir)
    path_sdf_result = gen_temp_file(suffix='.sdf', dir=cur_tmp_output_dir)
    command = './AutoDock-Vina-1.2.3/vina'
    arguments = [
        '--receptor', args.receptor,
        '--ligand', ligand,
        '--cpu', 1,
        '--seed', 42,
        '--out', tmp_pdbqt_docked,
        '--center_x', args.box['center_x'],
        '--center_y', args.box['center_y'],
        '--center_z', args.box['center_z'],
        '--size_x', args.box['size_x'],
        '--size_y', args.box['size_y'],
        '--size_z', args.box['size_z'],
        '--exhaustiveness', args.exhaustiveness,
        '--num_modes', args.number_poses
    ]
    result = execute_command(command, arguments, check=True)
    docking_flag = wait_until_file_generation(tmp_pdbqt_docked)
    # path_sdf_result = os.path.join(args.output_dir,  f"out_vina_docking_{args.filename}_{ligand_name}_{args.job_id}.sdf")
    obabel_arguments = [tmp_pdbqt_docked, '-ipdbqt', '-osdf', '-O', path_sdf_result]
    obabel_result = execute_command('obabel', obabel_arguments, check=False)
    obabel_flag = wait_until_file_generation(path_sdf_result)
    result = integrate_result(path_sdf_result, mol_name, ligand_name)
    return result


def main(args):
    #step 1
    temp_target_pdbqt, box = target_preparator(args.receptor_pdb_path, args.reference_ligand_path)
    args.receptor, args.box = temp_target_pdbqt, box
    #step 2 HTVS docking
    ligands = glob(args.filename+'/*.pdbqt')
    func0 = partial(docking_job, args=args)
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK')) if os.getenv('SLURM_CPUS_PER_TASK') else cpu_count()
    print('cpus_per_task', cpus_per_task)
    print('cpu_count()', cpu_count())
    print('------vina docking-------')
    with Pool(cpus_per_task) as pool:
        results = pool.map(func0, ligands)
    results = list(filter(None, results))
    # for ligand in ligands:
    #     results = docking_job(ligand, args=args)
    #step3 get results
    results_df = pd.DataFrame(results, columns=['conformer_name', 'ligand_name', 'conformer_idx', 'score', 'smiles', 'lowest_conformer', 'mol'])
    results_df.sort_values('score', ascending=True, inplace=True, ignore_index=True)
    topk = args.topk if isinstance(args.topk, int) else math.floor(len(results_df)*args.topk)
    results_df = results_df[:topk]
    #step 6
    with Chem.SDWriter(os.path.join(args.output_dir, f"out_vina_docking_{args.filename}_{args.job_id}.sdf"))as f:
        for idx, row in results_df.iterrows():
            mol = row['mol']
            mol.SetProp("_Name", row['conformer_name'])
            mol.SetProp('SCORE', str(row['score']))
            f.write(mol)
    results_df.drop(columns=['mol', 'conformer_idx', 'lowest_conformer'], inplace=True)
    results_df.to_csv(os.path.join(args.output_dir, f"out_vina_docking_{args.outfilename}_{args.job_id}.csv"), index=False)
    # with open(os.path.join(args.output_dir, f"out_vina_docking_{args.outfilename}_{args.job_id}.pkl"), 'wb')as f:
    #     pickle.dump(results_df, f)

    os.remove(temp_target_pdbqt)
    print(results_df.head(10))
    return results_df



if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser(description="vina docking")
    parser.add_argument("--config_file", type=str, help="Path to the JSON file containing parameters.",
                        default="/mnt/home/linjie/projects/AIDD/aidd_project/configs/linjie/vina_docking/0.json")
    parser.add_argument("-j", "--job_id", type=str, help="slurm job id", default='123456')
    parser.add_argument("-o", "--output_dir", type=str, help="Output dir of result", default='./outputs')
    parser.add_argument("-fn", "--filename", type=str, help="database name", default='./data/vina_docking/data/BAAAHL')
    args = parser.parse_args()
    args = format_args(args)
    main(args)
    print(f"python finished run time:{round(time() - start_time, 5)} seconds")



