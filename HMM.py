#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_hhsuite_installed():
    try:
        subprocess.run(['hhblits', '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def check_database(db_path):

    db_dir = Path(db_path)
    required_files = ['_hhm', '_cs219', '_a3m', '_ffindex']
    
    if db_dir.is_file():
        db_base = str(db_dir)
    else:
        db_base = str(db_dir / 'uniclust30')
    
    for ext in required_files:
        if not any(Path(db_base + ext).exists() for ext in ['.ffdata', '.ffindex']):
            return False
    return True

def run_hhblits(input_fasta, output_dir, database, cpu=4, iterations=2, e_value=0.001):

    os.makedirs(output_dir, exist_ok=True)

    basename = Path(input_fasta).stem
    a3m_file = os.path.join(output_dir, f"{basename}.a3m")
    hhr_file = os.path.join(output_dir, f"{basename}.hhr")

    cmd = [
        'hhblits',
        '-i', input_fasta,
        '-d', database,
        '-oa3m', a3m_file,
        '-o', hhr_file,
        '-n', str(iterations),
        '-e', str(e_value),
        '-cpu', str(cpu)
    ]


def run_hhmake(a3m_file, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    
    basename = Path(a3m_file).stem
    hmm_file = os.path.join(output_dir, f"{basename}.hmm")
    
    # hhmake命令
    cmd = [
        'hhmake',
        '-i', a3m_file,
        '-o', hmm_file
    ]
    

def process_single_sequence(fasta_file, db_path, output_dir, cpu=4):


    a3m_file = run_hhblits(fasta_file, output_dir, db_path, cpu)
    if not a3m_file:
        return None

    hmm_file = run_hhmake(a3m_file, output_dir)
    
    return hmm_file

def main():
    parser = argparse.ArgumentParser(description='use HH-suite get HMM feature')
    parser.add_argument('-i', '--input', required=True, 
                       help='input FASTA file')
    parser.add_argument('-d', '--database', required=True,
                       help='Uniclust30 dataset path (/path/to/uniclust30/uniclust30)')
    parser.add_argument('-o', '--output', default='./hmm_output',
                       help='output path (such as: ./hmm_output)')
    parser.add_argument('--cpu', type=int, default=4,
                       help='CPU (such as: 4)')
    parser.add_argument('--iterations', type=int, default=2,
                       help='hhblits iters (such as: 2)')
    
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        hmm_file = process_single_sequence(str(input_path), args.database, 
                                          args.output, args.cpu)
        if hmm_file:
            print(f"HMM finish: {hmm_file}")
    elif input_path.is_dir():

        fasta_files = list(input_path.glob('*.fasta')) + list(input_path.glob('*.fa'))
        for fasta_file in fasta_files:
            hmm_file = process_single_sequence(str(fasta_file), args.database,
                                              args.output, args.cpu)


if __name__ == '__main__':
    main()

#python HMM.py -i protein.fasta -d /path/to/uniclust30/uniclust30 -o ./hmm_output    