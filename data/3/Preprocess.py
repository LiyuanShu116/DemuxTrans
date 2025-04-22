import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os
from collections import Counter
from utils.Quipu.tools import normaliseLength

directory = '../3/'

feather_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.feather')]

combined_data_1 = pd.DataFrame()

for feather_file in feather_files:
    data = pd.read_feather(feather_file)
    combined_data_1 = pd.concat([combined_data_1, data], ignore_index=True)

data = combined_data_1[['reference_kmer','pA_signal']]

del combined_data_1

reference_kmers = [
    'TGCTG', 'TGTGT', 'GCCCC', 'GCCGC', 'CCCTC', 'TCCGC', 'CCTGG', 'CACTT',
    'CCTGC', 'ATGGT', 'AGAGA', 'ATCCC', 'TGGGA', 'AGGCA', 'GCGGG', 'CCCCC',
    'CGCCG', 'TTTCC', 'GGGAG', 'AGCTT', 'GCGGC', 'ATTGT', 'TTTTG', 'AGAGG',
    'AGAAG', 'CCCGC', 'TGTAA', 'GCAGG', 'CTGTC', 'GGGCT', 'AGTGA', 'CACCC',
    'GGCGG', 'CCCCA', 'ACACA', 'GCAGA', 'GCCTG', 'CCAGC', 'TGCTC', 'AGCTA',
    'GGGGC', 'CCGCC', 'TGCCT', 'TCCCT', 'GGCCG', 'TATCT', 'TTCCT', 'TCTGC',
    'GGAGG', 'GAGGC', 'CAGAG', 'GCTGT', 'AGGCT', 'CCCAG', 'CCATT', 'TGGGG',
    'CAGGC', 'AGAAA', 'ACAGA', 'GCCCT', 'GGGTG', 'AATTG', 'TTTTT', 'GTTTT',
    'TGGAG', 'TTCTC', 'CTACA', 'GGGAA', 'ACCCT', 'AAGCG', 'GGAAG', 'CTTTG',
    'CCCTG', 'AGGGC', 'AGGCC', 'CAGCC', 'AGCCT', 'AGCTG', 'CTGGA', 'CCTCC',
    'CTCCG', 'TCCCA', 'TTGGG', 'ACAGC', 'GGGCC', 'TGGCT', 'CCCAC', 'GCAGC',
    'GATTC', 'GGCAG', 'GTGTA', 'GCTGC', 'CTCTC', 'CCCGG', 'GCGCG', 'TCTAC',
    'CTTGG', 'TCTGT', 'TGTCC', 'GGCCC', 'AGGTA', 'TGTTT', 'CCTTT', 'GAGGA',
    'TGCCC', 'ACCAT', 'AGAGC', 'AGGAG', 'CCCCG', 'GGGCG', 'GTTCT', 'GGGGG',
    'AAAGA', 'TTGAC', 'AGCAC', 'GCCAC', 'CCGGC', 'GGTCT', 'TTAGG', 'CAGGA',
    'TGCAC', 'TTCGA', 'AAATA', 'CTGCC', 'AGGGG', 'TGCTT', 'TTGGC', 'CACGC',
    'TTCCC', 'CCCCT', 'CGGCC', 'GCTGG', 'GTTTA', 'ATGAA', 'CGCCC', 'ATGGG',
    'TAAAA', 'AATGA', 'TCTTT', 'TCTGG', 'TCCAC', 'GTGGG', 'CTCTT', 'CAAGT',
    'AGCCA', 'TCAGC', 'CTATC', 'AACCC', 'GTTGG', 'CTGTT', 'GGTAG', 'GGTGA',
    'GTCCC', 'GCCCG', 'CCAGG', 'GTGGC', 'TGTCA', 'AACCT', 'GCAAG', 'CAATA',
    'ATGCA', 'CTCCC', 'TGTGG', 'AGCAG', 'GTAGG', 'ATGGA', 'TGAGT', 'GAGAG',
    'TGAGC', 'GAAGG', 'TTCCA', 'CTCCT', 'ATGAG', 'TACAA', 'TGCCA', 'TTCTG',
    'AGCTC', 'AATGC', 'CGCAC', 'CTGGC', 'CGCAG', 'TCCCC', 'TGTGA', 'GGAGC',
    'GGCCT', 'AGGAC', 'TCAAC', 'CCTTC', 'TCTCC', 'GTAGA', 'TGGCG', 'CGGAC',
    'AGGAA', 'GGCCA', 'GGCGC', 'GGGGT', 'CTGCT', 'AATCT', 'CCCAT', 'CTTCC',
    'CTGAG', 'TTCAA', 'AGTTC', 'TGTTG', 'TTTTC', 'ATAGC', 'GGGGA', 'GGCTC',
    'GCGTC', 'CACTG', 'AGGCG', 'TAAAC', 'CATTG', 'TCGCT', 'CGAGC', 'AGATG',
    'TGTTC', 'AGTGG', 'GGACA', 'CTTCG', 'GCCAT', 'CACCA', 'CCTCT', 'TTTGG',
    'TCTCT', 'TCCAG', 'GTGTC', 'AAGGC', 'CCACC', 'ATTGC', 'TAGTG', 'CCCTT',
    'CGCTG', 'GAGTG', 'GACAG', 'AATGG', 'CCACT', 'CTTGC', 'GGCTG', 'CACAC',
    'GAATC', 'AACAT', 'AACTT', 'AAGAA', 'GGGAC', 'TGCAG', 'GGTGG', 'TTCAG',
    'CCGGG', 'CGGGC', 'CACCT', 'TGGCA', 'TGCGT', 'ACATC', 'CCAAA', 'TTCAC',
    'TTAAG', 'TGAGG', 'ATCAT', 'CATGG', 'AGACT', 'AAGTC', 'AAACA', 'AGGGA',
    'CTTCT', 'CCTGT', 'TTTGC', 'CACTC', 'TTGAA', 'TTGAG', 'ACCAC', 'TGGGC',
    'CGAGT', 'CAGCT', 'TGGCC', 'GTGTG', 'GAGGG', 'ACTTT', 'GGTTC', 'AGGAT',
    'TACTT', 'TCACC', 'AAATG', 'ATTTC', 'GCCAG', 'GCACC', 'GGAGA', 'ACCAG',
    'AACGC', 'TAGAT', 'TCCAA', 'AAAAA', 'GGCAC', 'CCGGT', 'CGCAA', 'CAGCG',
    'TGAAA', 'TCTTG', 'AACTG', 'TAATA', 'CGGCG', 'AAGCC', 'GAGCC', 'CAGGG',
    'ACTGA', 'TGCGG', 'CTCGC', 'TTTGT', 'AAAGT', 'CAGCA', 'CGGGG', 'GGGCA',
    'GTGAT', 'CTGTG', 'ATGTG', 'GCTTG', 'GTAAT', 'ACAGT', 'TCCTA', 'CCCGA',
    'CCTGA', 'CATGT', 'ACCCC', 'GGACG', 'GTGAG', 'CTTTC', 'GATGA', 'TAGGT',
    'CAACT', 'TAGCA', 'CTCTG', 'TTGCT', 'AGAGT', 'GACCT', 'TATTT', 'CTGGG',
    'GAAAA', 'AATAG', 'CCATC', 'AGCCC', 'TTGCC', 'TCTTA', 'GAAGC', 'CCTAC',
    'TCAGA', 'CCTCA', 'ACACC', 'GGTAT', 'GCCCA', 'TCCTG', 'ATGTT', 'AACCA',
    'CTGAC', 'CCGTG', 'CCGCA', 'CCAGT', 'TGGGT', 'GATGT', 'AGCCG', 'CTCCA',
    'TGACG', 'ACGGC', 'CCCAA', 'AAGGG', 'TCATA', 'TTATT', 'TCGTT', 'AACAG',
    'CAAGC', 'GGTTT', 'TGATG', 'AGTGC', 'AAAGG', 'CTTGA', 'GGTGT', 'TCGAC',
    'AAGAG', 'CCTTG', 'GCACG', 'TGTCT', 'AACTC', 'ATCTG', 'TACGG', 'ATTAG',
    'GTGCC', 'CAATG', 'TCAAT', 'AGTCT', 'GTGTT', 'GCGGT', 'GCAGT', 'AAGGA',
    'ACCAA', 'GGATT', 'AGGGT', 'GCCAA', 'AGGTC', 'TCGTA', 'ATCTC', 'GGCTT'
]

barcode = []

for i in range(400):
    barcode[i] = data[data['reference_kmer'] == reference_kmers[i]].head(3540)











