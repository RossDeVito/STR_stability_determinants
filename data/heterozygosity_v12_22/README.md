Data on STR variation in the sample population.

StatSTR output data used to label STRs should be in this dir. The command to run statSTR should be something like:

	statSTR --vcf /projects/ps-gymreklab/helia/ensembl/ensemble_out/all/merged_all.vcf.gz --out statSTR_merged_all --vcftype hipstr --het --entropy --numcalled --var --mean --mode --acount --afreq > run.out

TODO: add what extra args are actually needed/used