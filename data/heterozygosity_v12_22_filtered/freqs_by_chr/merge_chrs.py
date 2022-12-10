import pandas as pd


if __name__ == '__main__':
	all_dfs = [
		pd.read_csv(f'freqs_chr{i}.tab', sep='\t', header=0) for i in range(1, 23)
	]
	merged_df = pd.concat(all_dfs, axis=0)
	merged_df.to_csv('../freqs_merged.tab', sep='\t', index=False)