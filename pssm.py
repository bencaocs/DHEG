from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle



def command_pssm(content, output_file,pssm_file):
    os.system('./ncbi-blast-2.2.28+/bin/psiblast \
                -query %s \
                -db ./database/blast_data/ \
                -num_iterations 3 \
                -out %s \
                -out_ascii_pssm %s &' %(content, output_file,pssm_file))

def pssm(proseq,outdir):
    inputfile = open(proseq,'r')
    content = ''
    input_file = ''
    output_file = ''
    pssm_file = ''
    chain_name = []
    for eachline in inputfile:
        if '>' in eachline:
            if len(content):
                temp_file = open(outdir + '/fasta/' + chain_name,'w')
                temp_file.write(content)
                input_file = outdir + '/fasta/' + chain_name
                output_file = outdir + '/' + chain_name + '.out'
                pssm_file = outdir + '/' + chain_name + '.pssm'                
                command_pssm(input_file, output_file,pssm_file)
                temp_file.close
            content = ''
            chain_name = eachline[1:5] + eachline[6:7]
        content +=  ''.join(eachline)
        #print content
        #print chain_name
    if len(content):
        temp_file = open(outdir + '/fasta/' + chain_name,'w')
        temp_file.write(content)
        input_file = outdir + '/fasta/' + chain_name
        output_file = outdir + '/' + chain_name + '.out'
        pssm_file = outdir + '/' + chain_name + '.pssm'
        command_pssm(input_file, output_file,pssm_file)  
        temp_file.close
    inputfile.close()  

def formateachline(eachline):
    col = eachline[0:5].strip()
    col += '\t' + eachline[5:8].strip()    
    begin = 9 
    end = begin +3
    for i in range(20):     
        begin = begin
        end = begin + 3
        col += '\t' + eachline[begin:end].strip()
        begin = end
    col += '\n'
    return col

if __name__ == '__main__':
     
    proseq = '/ifs/home/liudiwei/experiment/step2/data/protein.seq'
    outdir = '/ifs/home/liudiwei/experiment/step2/pssm'
    pssm(proseq,outdir)
       

