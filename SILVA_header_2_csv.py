#Reading the whole SILVA database header as a panda dataframe to import the complete taxonomy

import pandas as pd
from Bio import Seq, SeqIO
from Bio.Alphabet import generic_dna
import sys

def fasta_2_df(SILVA_path):
	reads=[]
	for record in SeqIO.parse(SILVA_path, "fasta"):
	    str_ = str(record.description).split(" ", 1)[1]
	    id_ = str(record.name).split(' ')[0]
	    if str_.count(';') == 6:
	        [kingdom, phylum, class_,order,family,genus,species] = str_.split(';')
	    elif str_.count(';') > 6:    
	        list =  str_.split(';')
	        [kingdom,phylum, class_,order,family,genus,species] = list[-7:]
	    else:
	        list =  str_.split(';')
	        num = len(list)
	        [kingdom,phylum, class_,order,family,genus,species][:num] = list
	        [kingdom,phylum, class_,order,family,genus,species][num:] = '_'*(7-num)
	    reads.append([id_,kingdom ,phylum, class_,order,family,genus,species])    
	    
	SILVA_header =pd.DataFrame(reads,columns=['id','kingdom','phylum','class_','order','family','genus','species'])
	return SILVA_header

def main():
	script = sys.argv[0]
	SILVA_path = sys.argv[1]
	SILVA_header =  fasta_2_df(SILVA_path)
	SILVA_header.to_csv('SILVA_header_All_Taxa.csv')

main()