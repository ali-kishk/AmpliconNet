python SILVA_header_2_csv.py SILVA_132_SSURef_tax_silva.fasta

python preprocess.py V2_SILVA.fa SILVA_header_All_Taxa.csv V2

python word2vec.py V2 6 320

python train.py V2 6 320 best_only

python evaluate.py V2 6 320 best_only


