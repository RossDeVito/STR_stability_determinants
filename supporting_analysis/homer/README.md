# all T
findMotifs.pl fasta_files/T_pre_0_50_fasta.fa fasta homer_output/T/pre_50_0_nogcweight -fasta fasta_files/T_pre_1_50_fasta.fa -p 8 -len 4,5,6,7,9,11 -noweight

findMotifs.pl fasta_files/T_pre_1_50_fasta.fa fasta homer_output/T/pre_50_1_nogcweight -fasta fasta_files/T_pre_0_50_fasta.fa -p 8 -len 4,5,7,9,11 -noweight

findMotifs.pl fasta_files/T_post_0_50_fasta.fa fasta homer_output/T/post_50_0_nogcweight -fasta fasta_files/T_post_1_50_fasta.fa -p 8 -len 4,5,6,7,9,11 -noweight

findMotifs.pl fasta_files/T_post_1_50_fasta.fa fasta homer_output/T/post_50_1_nogcweight -fasta fasta_files/T_post_0_50_fasta.fa -p 8 -len 4,5,6,7,9,11 -noweight

findMotifs.pl fasta_files/T_post_1_50_fasta.fa fasta homer_output/T/post_50_1 -fasta fasta_files/T_post_0_50_fasta.fa -p 8 -len 4,5,6,7,9,11

# CA all
findMotifs.pl fasta_files/CA_pre_1_6to15_50_fasta.fa fasta homer_output/CA/pre_50_1 -fasta fasta_files/CA_pre_0_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/CA_pre_0_6to15_50_fasta.fa fasta homer_output/CA/pre_50_0 -fasta fasta_files/CA_pre_1_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/CA_post_1_6to15_50_fasta.fa fasta homer_output/CA/post_50_1 -fasta fasta_files/CA_post_0_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/CA_post_0_6to15_50_fasta.fa fasta homer_output/CA/post_50_0 -fasta fasta_files/CA_post_1_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

# AC all
findMotifs.pl fasta_files/AC_pre_1_6to15_50_fasta.fa fasta homer_output/AC/pre_50_1 -fasta fasta_files/AC_pre_0_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/AC_pre_0_6to15_50_fasta.fa fasta homer_output/AC/pre_50_0 -fasta fasta_files/AC_pre_1_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/AC_post_1_6to15_50_fasta.fa fasta homer_output/AC/post_50_1 -fasta fasta_files/AC_post_0_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

findMotifs.pl fasta_files/AC_post_0_6to15_50_fasta.fa fasta homer_output/AC/post_50_0 -fasta fasta_files/AC_post_1_6to15_50_fasta.fa -p 8 -len 4,5,6,7,9,11

# AG pre
findMotifs.pl fasta_files/AG_pre_1_6to15_30_fasta.fa fasta homer_output/AG/pre_30_1 -fasta fasta_files/AG_pre_0_6to15_30_fasta.fa -p 8 -len 4,5,6,7,9,11

# CT pre
findMotifs.pl fasta_files/CT_pre_1_6to15_30_fasta.fa fasta homer_output/CT/pre_30_1 -fasta fasta_files/CT_pre_0_6to15_30_fasta.fa -p 8 -len 4,5,6,7,9,11

# AT pre
findMotifs.pl fasta_files/AT_pre_1_6to15_30_fasta.fa fasta homer_output/AT/pre_30_1 -fasta fasta_files/AT_pre_0_6to15_30_fasta.fa -p 8 -len 4,5,6,7,9,11

# TA pre
findMotifs.pl fasta_files/TA_pre_1_6to15_30_fasta.fa fasta homer_output/TA/pre_30_1 -fasta fasta_files/TA_pre_0_6to15_30_fasta.fa -p 8 -len 4,5,6,7,9,11

# GT pre
findMotifs.pl fasta_files/GT_pre_1_6to15_30_fasta.fa fasta homer_output/GT/pre_30_1 -fasta fasta_files/GT_pre_0_6to15_30_fasta.fa -p 8 -len 4,5,6,7,9,11
