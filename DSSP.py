#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_dssp_installed():

    try:

        subprocess.run(['mkdssp', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        try:
            subprocess.run(['dssp', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

def get_dssp_command():

    try:
        subprocess.run(['mkdssp', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'mkdssp'
    except FileNotFoundError:
        return 'dssp'

def run_dssp(pdb_file, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    basename = Path(pdb_file).stem
    dssp_file = os.path.join(output_dir, f"{basename}.dssp")
    
    dssp_cmd = get_dssp_command()
    
    # DSSP 
    cmd = [dssp_cmd, '-i', pdb_file, '-o', dssp_file]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return dssp_file
    except subprocess.CalledProcessError as e:
        print(f"DSSP filed: {e}")
        return None

def parse_dssp(dssp_file):

    secondary_structure = []
    accessibility = []
    
    try:
        with open(dssp_file, 'r') as f:
            lines = f.readlines()
            

        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('  #  RESIDUE'):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            if len(line) > 10:
                try:
                    ss_code = line[16] if len(line) > 16 else ' '
                    
                    if ss_code in ['H', 'G', 'I']:  # α-helix, 3-10 helix, π-helix
                        ss_simple = 'H'
                    elif ss_code in ['E', 'B']:     # β-strand, β-bridge
                        ss_simple = 'E'
                    else:
                        ss_simple = 'C'             # others
                  
                    acc = float(line[34:38].strip()) if line[34:38].strip() != '' else 0.0
                    
                    secondary_structure.append(ss_simple)
                    accessibility.append(acc)
                    
                except (ValueError, IndexError):
                    continue
                    
    except Exception as e:
        print(f"filed: {e}")
        return None, None
    
    return secondary_structure, accessibility

def save_secondary_structure(dssp_file, output_dir):

    basename = Path(dssp_file).stem
    ss_file = os.path.join(output_dir, f"{basename}_ss.txt")
    acc_file = os.path.join(output_dir, f"{basename}_acc.txt")
    
    ss_list, acc_list = parse_dssp(dssp_file)
    
    if ss_list and acc_list:
        with open(ss_file, 'w') as f:
            f.write(''.join(ss_list))
        with open(acc_file, 'w') as f:
            f.write('\n'.join(map(str, acc_list)))
    
        
        return ss_file, acc_file
    
    return None, None

def pdb_to_fasta(pdb_file, output_dir):

    basename = Path(pdb_file).stem
    fasta_file = os.path.join(output_dir, f"{basename}.fasta")
    
    sequence = []
    prev_resnum = None
    
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    resname = line[17:20].strip()
                    chain = line[21]
                    resnum = line[22:26].strip()
                    
                    if chain != ' ':
                        aa_map = {
                            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                            'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                            'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                            'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                            'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                        }
                        
                        if resname in aa_map and resnum != prev_resnum:
                            sequence.append(aa_map[resname])
                            prev_resnum = resnum
                            
    except Exception as e:
        print(f"filed: {e}")
        return None
    
    if sequence:
        with open(fasta_file, 'w') as f:
            f.write(f">{basename}\n")
            f.write(''.join(sequence))
        return fasta_file
    
    return None

def main():
    parser = argparse.ArgumentParser(description='ues DSSP get structure feature')
    parser.add_argument('-i', '--input', required=True,
                       help='pdb file or folder')
    parser.add_argument('-o', '--output', default='./dssp_output',
                       help='output path (such as: ./dssp_output)')
    parser.add_argument('--extract-seq', action='store_true',
                       help='get sequence to save FASTA')
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    

    input_path = Path(args.input)
    pdb_files = []
    
    if input_path.is_file():
        pdb_files.append(str(input_path))
    elif input_path.is_dir():
        pdb_files.extend([str(f) for f in input_path.glob('*.pdb')])
        pdb_files.extend([str(f) for f in input_path.glob('*.ent')])

    
    if not pdb_files:
        print("not find PDB file")
        sys.exit(1)
    
    for pdb_file in pdb_files:
        print(f"\n process PDB file: {pdb_file}")
        
        dssp_file = run_dssp(pdb_file, args.output)
        
        if dssp_file:
            save_secondary_structure(dssp_file, args.output)
            
            if args.extract_seq:
                fasta_file = pdb_to_fasta(pdb_file, args.output)
                if fasta_file:
                    print(f"sequence save as: {fasta_file}")

if __name__ == '__main__':
    main()

# python DSSP.py -i protein.pdb -o ./dssp_output    