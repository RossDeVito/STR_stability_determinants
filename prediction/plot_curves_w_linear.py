"""Plot PR and ROC curves for a set of models."""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	save_output = True
	# save_desc = '0_1_2_3_4_5_6_7_8_9_10_11_12_13_14' # if None will name with timestamp int
	# save_desc = '10_14_5_w_2lin'
	save_desc = 'best_w_and_wo_len'

	label_version = ['v1-mfr0_005_mnc2000-m6_5', 'v1-mfr0_0025_mnc2000-m7_5'][0]
	label_version_lin = ['mfr0_005_mnc2000-m50'][0]

	trained_res_dir = os.path.join('training_output', label_version)

	models_to_plot = [
		{
			'name': 'Best Blinded Model',
			'path': 'version_10',
			'which_res': 'best'
		},
		{
			'name': 'Best Model with STR Length',
			'path': 'version_20',
			'which_res': 'best'
		}
	]

	linestyle = ['-', '--', '-.', ':'][0]
	lw = 3

	linear_model_dir = os.path.join(
		'..', 'supporting_analysis', 'pattern_matching', 'linear_models',
		label_version_lin
	)
	linear_model_dirs = []#['default_log_reg', 'default_log_reg_no_len']

	# Load results
	results = []

	for mod in models_to_plot:
		res_dir = os.path.join(trained_res_dir, mod['path'], 'results')
		res_files = os.listdir(res_dir)
		if 'which_res' in mod and mod['which_res'] != 'all':
			if mod['which_res'] == 'best':
				res_files = [f for f in res_files if 'best' in f]
			elif mod['which_res'] == 'last':
				res_files = [f for f in res_files if 'last' in f]
			elif mod['which_res'] == 'mean':
				res_files = [f for f in res_files if 'mean' in f]
			elif mod['which_res'] == 'agree':
				res_files = [f for f in res_files if 'agree' in f]

		for fname in res_files:
			res_file = os.path.join(res_dir, fname)
			with open(res_file, 'r') as f:
				res = json.load(f)
				# if 'best' in fname:
				# 	res['name'] = mod['name'] + ' (best)'
				# elif 'last' in fname:
				# 	res['name'] = mod['name'] + ' (last)'
				# elif '_results_mean' in fname:
				# 	res['name'] = mod['name'] + ' (mean)'
				# elif '_results_vote' in fname:
				# 	res['name'] = mod['name'] + ' (voting)'
				# elif '_results_agree' in fname:
				# 	res['name'] = mod['name'] + ' (all agree)'
				# else:
				res['name'] = mod['name']
				results.append(res)

	# Load linear models
	for lm_dir in linear_model_dirs:
		res_json = os.path.join(linear_model_dir, lm_dir, 'results.json')
		with open(res_json, 'r') as f:
			res = json.load(f)
		res = res['test']
		res['name'] = lm_dir
		results.append(res)

	# plot ROC and PR curves
	if save_desc is None:
		time_int = int(datetime.now().timestamp()) # for naming plots
		print(time_int)
	else:
		time_int = save_desc # excuse this variable naming pls

	fig, (ax_roc, ax_pr_0, ax_pr_1) = plt.subplots(1, 3, sharex=True, sharey=True, 
										subplot_kw=dict(box_aspect=1),
										figsize=(14, 6))
	ax_roc.plot([0, 1], [0, 1], color="black", lw=lw, linestyle=":")

	for i,res in enumerate(results):
		ax_roc.plot(res['ROC_fpr'], res['ROC_tpr'], label=res['name'], lw=lw,
					linestyle=linestyle, zorder=len(results)-i)
		ax_pr_0.plot(res['PR_recall_0'], res['PR_prec_0'], label=res['name'], 
						lw=lw, linestyle=linestyle, zorder=len(results)-i)
		ax_pr_1.plot(res['PR_recall_1'], res['PR_prec_1'], label=res['name'], 
						lw=lw, linestyle=linestyle, zorder=len(results)-i)

	fig.suptitle("Binary Stability Prediction")
	ax_roc.set_title("ROC")
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.legend().set_visible(False)
	ax_pr_0.set_title("PR Curve Stable")
	ax_pr_0.set_xlabel("Recall")
	ax_pr_0.set_ylabel("Precision")
	ax_pr_0.legend(loc='lower left')
	ax_pr_1.set_title("PR Curve Unstable")
	ax_pr_1.set_xlabel("Recall")
	ax_pr_1.set_ylabel("Precision")
	ax_pr_1.legend().set_visible(False)
	fig.tight_layout()

	plt.show()
	if save_output:
		fig.savefig(
			os.path.join(trained_res_dir, 'roc_pr_{}.png'.format(time_int)),
			bbox_inches='tight'
		)
	plt.close(fig)

	# plot confusion matrices
	unit_len = 3
	fig, axs = plt.subplots(ncols=len(results), subplot_kw=dict(box_aspect=1),
				figsize=(len(results)*unit_len, unit_len))

	max_count = np.max([np.array(r['confusion_matrix']).max() for r in results])

	for i, res in enumerate(results):
		axs[i].set_title(res['name'])
		sns.heatmap(
			np.array(res['confusion_matrix']) / np.array(res['confusion_matrix']).sum(1, keepdims=True), 
			# res['confusion_matrix'],
			annot=res['confusion_matrix'],
			fmt='0.0f',
			cbar=False, 
			ax=axs[i],
			cmap='Purples',
			vmin=0,
			vmax=1.1,
			# center=.25,
			# cbar_ax=axs[-1]
		)

	# max_count = np.max([np.array(r['confusion_matrix']).max() for r in results])

	# for i, res in enumerate(results):
	# 	axs[i].set_title(res['name'])
	# 	sns.heatmap(
	# 		# np.abs(
	# 		# 	np.array(res['confusion_matrix']) 
	# 		# 	/ np.array(res['confusion_matrix']).sum(1, keepdims=True).repeat(2).reshape(2,2)
	# 		# 	- .5), 
	# 		res['confusion_matrix'],
	# 		annot=res['confusion_matrix'],
	# 		fmt='0.0f',
	# 		cbar=False, 
	# 		ax=axs[i],
	# 		cmap='Purples',
	# 		vmin=0,
	# 		vmax=max_count * 1.25,
	# 		# center=.25,
	# 		# cbar_ax=axs[-1]
	# 	)

	axs[0].set_ylabel("True")
	axs[0].set_xlabel("Pred")
	fig.tight_layout()
	plt.show()
	if save_output:
		fig.savefig(
			os.path.join(trained_res_dir, 'CMs_{}.png'.format(time_int)),
			bbox_inches='tight')
	plt.close(fig)

	# Make table
	df = pd.DataFrame(results)

	table = df[['name', 'macro_F1', 'ROC_auc', 'class_precision', 'class_recall']]
	# table['macro_F1'] = [s.item() for s in table.macro_F1]
	# table['ROC_auc'] = [s.item() for s in table.ROC_auc]
	table['class_precision_0'] = [s[0] for s in table.class_precision]
	table['class_precision_1'] = [s[1] for s in table.class_precision]
	table['class_recall_0'] = [s[0] for s in table.class_recall]
	table['class_recall_1'] = [s[1] for s in table.class_recall]
	table = table.drop(columns=['class_precision', 'class_recall'])
	# Get avg rank
	table['Macro_F1_rank'] = table.macro_F1.rank(ascending=False)
	table['ROC_auc_rank'] = table.ROC_auc.rank(ascending=False)
	table['mean_rank'] = (table.Macro_F1_rank + table.ROC_auc_rank) / 2

	if save_output:
		table.to_csv(
			os.path.join(trained_res_dir, 'res_table_{}.csv'.format(time_int)),
		)

	print(table)

	print("Sorted by Macro F1:")
	print(table.sort_values(by='macro_F1', ascending=False))

	print("Sorted by ROC AUC:")
	print(table.sort_values(by='ROC_auc', ascending=False))

	print("Sorted by mean rank:")
	print(table.sort_values(by='mean_rank', ascending=True))


