import os
from functools import partial
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from p_tqdm import p_map
from tqdm import tqdm

from torchmetrics import MetricCollection
from torchmetrics import ConfusionMatrix, Precision, Recall, F1, PrecisionRecallCurve
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from find_patterns import find_patterns


def get_feat_val(samp, cases, d_vals, agg_method='sum'):
	pre_X = samp.str_seq[0]
	pre_Y = samp.str_seq[1]
	post_X = samp.str_seq[-1]
	post_Y = samp.str_seq[-2]

	# get feature counts
	pre_feats = find_patterns(np.array(list(reversed(samp.pre_seq))), pre_X, pre_Y, max_d=max(d_vals))
	post_feats = find_patterns(np.array(list(samp.post_seq)), post_X, post_Y, max_d=max(d_vals))

	if agg_method == 'sum':
		return [
			pre_feats[case][d] + post_feats[case][d] for case,d in zip(cases, d_vals)
		]
	elif agg_method == 'max':
		return [
			max(pre_feats[case][d], post_feats[case][d]) for case,d in zip(cases, d_vals)
		]

def score_model(y_true, y_pred):
	metrics = MetricCollection({
		'macro_precision': Precision(num_classes=2, average='macro', multiclass=True),
		'class_precision': Precision(num_classes=2, average='none', multiclass=True),
		'macro_recall': Recall(num_classes=2, average='macro', multiclass=True),
		'class_recall': Recall(num_classes=2, average='none', multiclass=True),
		'macro_F1': F1(num_classes=2, average='macro', multiclass=True),
		'class_F1': F1(num_classes=2, average='none', multiclass=True),
		'confusion_matrix': ConfusionMatrix(num_classes=2)
	})

	res_dict = metrics(torch.tensor(y_pred), torch.tensor(y_true))
	# make numpy, so can then be turned into a list before saving as JSON
	res_dict = {k: v.numpy() for k, v in res_dict.items()}
	res_dict['y_true'] = y_true
	res_dict['y_pred'] = y_pred

	res_dict['num_true_0'] = (y_true == 0).sum().item()
	res_dict['num_true_1'] = (y_true == 1).sum().item()

	res_dict['ROC_fpr'], res_dict['ROC_tpr'], _ = roc_curve(
		y_true, y_pred
	)
	res_dict['ROC_auc'] = auc(res_dict['ROC_fpr'], res_dict['ROC_tpr'])

	res_dict['PR_prec_1'], res_dict['PR_recall_1'], _ = precision_recall_curve(
		y_true, y_pred, pos_label=1
	)
	res_dict['PR_auc_1'] = auc(res_dict['PR_recall_1'], res_dict['PR_prec_1'])

	res_dict['PR_prec_0'], res_dict['PR_recall_0'], _ = precision_recall_curve(
		y_true, -y_pred, pos_label=0
	)
	res_dict['PR_auc_0'] = auc(res_dict['PR_recall_0'], res_dict['PR_prec_0'])

	# Save results as lists
	return {
		k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in res_dict.items() 
	}

if __name__ == '__main__':
	__spec__ = None

	# Options
	label_version = [
		'mfr0_005_mnc2000-m50',
		'mfr0_0025_mnc2000-m50',
		'mfr0_0_mnc2000-m64'
	][1]
	label_version_dir = os.path.join(
		'find_patterns_output',
		label_version
	)

	max_str_len = 15
	cases = [1, 2, 5]	# cases and d vals paired by index (e.i. 
	d_vals = [0, 1, 0]	#  zip(cases, d_vals) gives wanted feats)
	agg_method = 'sum'
	use_str_len = True

	inc_complements = False
	num_cpus = 10
	use_validation_set = False

	# Load data
	samp_df = pd.read_csv(os.path.join(label_version_dir, 'samples.csv'))
	if not inc_complements:
		samp_df = samp_df[samp_df.complement == False]
	samp_df = samp_df[samp_df.motif_len.values * samp_df.num_copies.values <= max_str_len]

	# generate additional features
	feat_cols = [
		'case{}_d{}_{}'.format(case, d, agg_method) for case,d in zip(cases, d_vals)
	]
	feat_vals = p_map(
		partial(get_feat_val, cases=cases, d_vals=d_vals, agg_method=agg_method),
		[samp for _,samp in samp_df.iterrows()],
		num_cpus=num_cpus
	)
	feats_df = pd.DataFrame(feat_vals, columns=feat_cols)
	if use_str_len:
		feats_df['STR_len'] = samp_df.num_copies.values * samp_df.motif_len.values
	
	# get splits and train model
	if use_validation_set:
		train_idx = np.where(samp_df.split_1 == 0)[0]
		val_idx = np.where(samp_df.split_1 == 1)[0]
	else:
		train_idx = np.where((samp_df.split_1 == 0) | (samp_df.split_1 == 1))[0]
	test_idx = np.where(samp_df.split_1 == 2)[0]

	X_train = feats_df.iloc[train_idx]
	y_train = samp_df.iloc[train_idx].label
	classifier = LogisticRegression().fit(X_train, y_train)

	# score
	scores = dict()

	y_true = y_train.values
	y_pred = classifier.predict_proba(X_train)[:, 1]
	scores['train'] = score_model(y_true, y_pred)

	if use_validation_set:
		y_true = samp_df.iloc[val_idx].label.values
		y_pred = classifier.predict_proba(feats_df.iloc[val_idx])[:, 1]
		scores['val'] = score_model(y_true, y_pred)

	y_true = samp_df.iloc[test_idx].label.values
	y_pred = classifier.predict_proba(feats_df.iloc[test_idx])[:, 1]
	scores['test'] = score_model(y_true, y_pred)

	# display scores of interest
	score_to_print = ['macro_F1', 'ROC_auc', 'class_F1', 'confusion_matrix']
	for score_name in score_to_print:
		print(score_name)
		print({k:s[score_name] for k,s in scores.items()})

	# Save and display coefficients
	scores['coef'] = {
		'coef': list(classifier.coef_[0]),
		'feat_names': list(feats_df.columns.values.astype(str))
	}
	print(scores['coef'])
	scores['meta'] = {
		'label_version': label_version,
		'inc_complements': inc_complements,
		'max_str_len': max_str_len,
		'cases': cases,
		'd_vals': d_vals,
		'agg_method': agg_method,
		'use_validation_set': use_validation_set,
	}

	# Save results
	model_desc = 'default_log_reg'
	res_save_dir = os.path.join('linear_models', label_version, model_desc)
	if not os.path.exists(res_save_dir):
		os.makedirs(res_save_dir)
	res_save_file = os.path.join(res_save_dir, 'results.json')
	with open(res_save_file, 'w') as f:
		json.dump(scores, f, indent=2)
