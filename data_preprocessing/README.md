# Reapeat Number Variation Data Preprocessing Pipeline
- find_target_STRs.py - Extracts STRs of the desired STR motif types from a hipstr_reference.bed.gz for the desired chromosome build. Requires pyfaidx library. Uses hipSTR reference bed file and reference genome.

- label_pure_STRs.py - Subsets unlabeled STRs to those where the reference is a pure repeat, then labels each STR with data from statSTR_output.tab. Outputs to labeled_samples_{desc}.json

- vis_labeled_data.py - Visualizes the labeled data number called, heterozygosity and entropy score distributions, and counts by motif to help decide on cutoffs values for the final preprocessing script.

- preprocess_STRs.py - Filter by STR type, max STR len, min num called. Labels samples wrt heterozygosity score thresholds. Saves to sample_data_{params}.json to use with dataloaders for training/other analysis.