# AmpliconNet: Sequence Based Multi-layer Perceptron for Amplicon Read Classification Using Real-time Data Augmentation 
[International Conference on Bioinformatics and Biomedicine 2018 Proceedings "In Press"]

16s rRNA neural network classifier using the direct sequence.

Taxonomic assignment is the core of targeted metagenomics approaches that aims to assign sequencing reads to
their corresponding taxonomy. Sequence similarity searching and machine learning (ML) are two commonly used approaches for
taxonomic assignment based on the 16S rRNA. Similarity based approaches require high computation resources, while ML
approaches don’t need these resources in prediction. The majority of these ML approaches depend on k-mer frequency rather than
direct sequence, which leads to low accuracy on short reads as k-mer frequency doesn’t consider k-mer position. Moreover training
ML taxonomic classifiers depend on a specific read length which may reduce the prediction performance by decreasing read length.
In this study, we built a neural network classifier for 16S rRNA reads based on SILVA database (version 132). Modeling was
performed on direct sequences using Convolutional neural network (CNN) and other neural network architectures such as Multi-layer
Perceptron and Recurrent Neural Network. In order to reduce modeling time of the direct sequences, In-silico PCR was applied
on SILVA database. Total number of 14 subset databases were generated by universal primers for each single or paired high
variable region (HVR). Moreover, in this study, we illustrate the results for the V2 database model on ~ 1850 classes on the genus
level. In order to simulate sequencing fragmentation, we trained variable length subsequences from 50 bases till the full length of
the HVR that are randomly changing in each training iteration.
Simple MLP model with global max pooling gives 0.93 test accuracy for the genus level (for reads of 100 base sub-sequences)
and 0.96 accuracy for the genus level respectively (on the full length V2 HVR). In this study, we present a novel method
AmpliconNet to model the direct amplicon sequence using MLP over a sequence of k-mers faster 20 times than CNN in training
and 10 times in prediction.

# Usage

## In prediction

```bash
python src/predict --dir_path INPUT_FASTQ_DIR  --database V46 --output_dir OUTPUT_DIRECTORY --model_path V46/models/MLP_SK.hdfs

```
Database have to be changed according the HVR primersof the study

# Building a taxonomy table

```bash
python src/Predict_Taxonomy_Table.py --pred_dir PREV_PRED_DIR --target_rank genus --o-taxa_table AmpliconNet_taxa_table.csv
```

# Training a new model

```bash
python src/SILVA_header_2_csv.py --silva_path SILVA_132_SSURef_tax_silva.fasta  --silva_header SILVA_header_All_Taxa.csv

python src/preprocess.py --hvr_database V2_SILVA.fa --silva_header SILVA_header_All_Taxa.csv --output_dir V2

python src/train.py --database_dir V2 --kmer_size 6  --batch_size 250 --training_mode mlp_sk

python src/evaluate.py --database_dir V2 --kmer_size 6 --batch_size 250  --training_mode mlp_sk
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
