sed -e 's/U/T/g' SILVA_132_SSURef_tax_silva.fasta > SILVA_132_SSURef_tax_silva_T_by_U.fasta
mkdir PCR_SILVA_all

ipcress Insilico_PCR/HVR_ipcress_extended.tsv SILVA_132_SSURef_tax_silva_T_by_U.fasta -P TRUE -m 0 > Insilico_PCR/PCR_extended_SILVA_132_SSURef_all.fasta 

cd Insilico_PCR

cat PCR_extended_SILVA_132_SSURef_all.fasta |grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product"|grep "^>" -a10 > PCR_extended_SILVA_132_sequence_only.fasta


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V2_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product"|grep -v "Matches" > V2_SILVA.fa

cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V3_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target"| grep -v "Product" |grep -v "Matches" > V3_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V4_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target"| grep -v "Product" |grep -v "Matches" > V4_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V5_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V5_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V7_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V7_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V8_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V8_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V23_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target"| grep -v "Product"  |grep -v "Matches"> V23_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V34_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V34_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V35_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product"|grep -v "Matches" > V35_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V45_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product"|grep -v "Matches" > V45_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V56_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product"|grep -v "Matches" > V56_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V67_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V67_SILVA.fa


cat PCR_extended_SILVA_132_SSURef_all.fasta |grep "^>V78_" -a10|grep -v "^-"|grep -v "^5"|grep -v "pcress"|grep -v "^Result" |grep -v "Experiment"|grep -v "|||"| grep -v "#"| grep -v "Primers"| grep -v "Target" | grep -v "Product" |grep -v "Matches"> V78_SILVA.fa



