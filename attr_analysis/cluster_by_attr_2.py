"""Cluster by attributions in a way that allows for comparison of clusters 
by label or TP/TN status.
"""

import os
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, dbscan
import hdbscan

import logomaker


def reformat_attribution_pkl(attr_data):
	"""Reformat attribution data from as read from pkl to numpy arrays
	or pandas dataframes.
	"""
	attr_data['pre'] = np.stack(attr_data['pre'])
	attr_data['post'] = np.stack(attr_data['post'])
	attr_data['pre_strings'] = np.stack(attr_data['pre_strings'])
	attr_data['post_strings'] = np.stack(attr_data['post_strings'])

	# fix a typo
	if 'predicitons' in attr_data.keys():
		attr_data['predictions'] = attr_data.pop('predicitons')

	attr_data['predictions'] = np.array(attr_data['predictions'])
	attr_data['labels'] = np.array(attr_data['labels'])
	attr_data['sample_data'] = pd.DataFrame(attr_data['sample_data'])

	return attr_data


def add_prediction_features(attr_data, cutoff=.5):
	"""Add binary prediction features and their correctness to attr_data."""
	attr_data['binary_pred'] = (attr_data['predictions'] > cutoff).astype(int)
	attr_data['correct_pred'] = (attr_data['binary_pred'] == attr_data['labels'])
	attr_data['true_pos'] = (attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
	attr_data['true_neg'] = (attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
	return attr_data


def subset_attr_data(attr_data, subset_mask):
	attr_data = copy.deepcopy(attr_data)

	for k,v in attr_data.items():
		if isinstance(v, np.ndarray):
			attr_data[k] = v[subset_mask]
		elif isinstance(v, pd.DataFrame):
			attr_data[k] = v.loc[subset_mask]

	return attr_data


def plot_cluster_attn_and_ppm(clust_labels, subset_data, n_per_side,
		str_pad_size, plt_height=1.0, plt_width=6.5, label_desc=''):
	"""Plot cluster attribution and ppm for each cluster.

	TODO: handle post STR seqs
	
	Args:
		clust_labels: array of cluster labels
		subset_data: attribution data in the normal dict form
		n_per_side: number of samples per side STR used
		str_pad_size: size of STR padding
		plt_height: height of total plot is plt_height * num_clusters
	"""
	sorted_labels = np.sort(np.unique(clust_labels))

	fig, axs = plt.subplots(len(sorted_labels), 2, 
		figsize=(plt_width,len(sorted_labels) * plt_height + int(len(sorted_labels)==1)*.25), 
		squeeze=False,
		sharex=True, 
		sharey='col'
	)

	# fig.suptitle("Logo before {} {}".format(start_pattern, label_desc))
	axs[0,0].set_title('Median Attribution Score')
	axs[0,1].set_title('Position Probability Matrix')

	max_attr_mag = 0
	min_attr_mag = 0
	
	for i,cluster in enumerate(sorted_labels):
		clust_mask = clust_labels == cluster
		clust_strings = np.array(
			[np.array(list(c[-n_per_side-str_pad_size:-str_pad_size])) for c in subset_data['pre_strings'][clust_mask]]
		)

		# get median attributions
		mean_ignore_0 = True
		if mean_ignore_0:
			full_attrs = subset_data['pre'][clust_mask][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
			full_attrs[full_attrs == 0] = np.nan
			with np.errstate(all='ignore'):
				med_attrs = np.nan_to_num(np.nanmedian(full_attrs, 0))
		else:
			med_attrs = subset_data['pre'][clust_mask][
				:,:4:,-n_per_side-str_pad_size:-str_pad_size].median(0)
		med_attr_df = pd.DataFrame(
			med_attrs.T, 
			columns=['A', 'C', 'G', 'T']
		)

		# For scaling plot
		max_attr_mag = max(
			max_attr_mag, 
			np.where(med_attr_df.values > 0, med_attr_df.values, 0).sum(1).max()
		)
		min_attr_mag = min(
			min_attr_mag, 
			np.where(med_attr_df.values < 0, med_attr_df.values, 0).sum(1).min()
		)

		logomaker.Logo(med_attr_df, ax=axs[i, 0], center_values=True)
		axs[i, 0].set_ybound(min_attr_mag, max_attr_mag)

		# get position probability matrix
		counts_mat = np.stack([
			(clust_strings == 'A').sum(0),
			(clust_strings == 'C').sum(0),
			(clust_strings == 'G').sum(0),
			(clust_strings == 'T').sum(0)
		])
		ppm = counts_mat / counts_mat.sum(0, keepdims=True)
		ppm_df = pd.DataFrame(ppm.T, columns=['A', 'C', 'G', 'T'])

		logomaker.Logo(ppm_df, ax=axs[i, 1])
		axs[i, 1].set_yticklabels([])

	# Format plot
	plt.setp(
		axs, 
		xticks=np.array(list(range(0, ppm.shape[1], 5))), 
		xticklabels=list(range(-ppm.shape[1], 0, 5))
	)
	min_y = min(a.dataLim.get_points()[0,1] for a in axs[:,0])
	max_y = max(a.dataLim.get_points()[1,1] for a in axs[:,0])
	plt.setp(axs[:,0], ybound=(min_y, max_y))

	plt.tight_layout(w_pad=.21, h_pad=.1)

	return fig, axs, sorted_labels


if __name__ == '__main__':
	"""Args:

		str_motif_len: length of STR motif(s) (e.g. 2 for {'CA', 'TG', 
			'GA'} motifs)
		str_pad_size: number of positions of attributions that are STR padding.
		n_per_side: number of loci to use in clustering.
		use_TP_TN: whether to use TP/TN status (True) or label (False) when
			seperating groups before clustering.
	"""
	# Options
	attrs_dir = '../prediction/training_output'
	label_version = [
		'v1-mfr0_005_mnc2000-m6_5',
		'v1-mfr0_0025_mnc2000-m7_5'
	][0]
	model_version = 'tscc_version_0'
	attr_file = 'ig_global_val_test.pkl'

	str_motif_len = 2
	str_pad_size = 4

	n_per_side = 15

	use_TP_TN = False

	cluster_metric = 'l1'
	cluster_method = 'hdbscan'
	dbscan_params = {
		'eps': .2,
		'min_samples': 40,
	}
	hdbscan_params = {
		'min_cluster_size': 10,
		'min_samples': 3,
		'cluster_selection_epsilon': 0.0,
		'alpha': 1.
	}

	clip_attrs = True
	clip_thresh = [10, 90]

	save_plots = True
	show_plots = False

	# Load single attr type
	attr_data = pd.read_pickle(os.path.join(
		attrs_dir, label_version, model_version, 'attributions', attr_file
	))

	# Reformat data and add prediction features
	attr_data = reformat_attribution_pkl(attr_data)
	attr_data = add_prediction_features(attr_data)
	attr_data['sample_data']['alpha_motif'] = attr_data['sample_data'].motif.apply(
		lambda x: ''.join(sorted([*x])) # motif name with letters sorted alphabetically
	)

	if use_TP_TN:
		pos_class_data = subset_attr_data(
			attr_data, 
			(attr_data['binary_pred'] == 1) & (attr_data['labels'] == 1)
		)
		neg_class_data = subset_attr_data(
			attr_data, 
			(attr_data['binary_pred'] == 0) & (attr_data['labels'] == 0)
		)
	else:
		pos_class_data = subset_attr_data(
			attr_data, 
			(attr_data['labels'] == 1)
		)
		neg_class_data = subset_attr_data(
			attr_data, 
			(attr_data['labels'] == 0)
		)

	# Add data on start/end of STR, since clustering will be done 
	#	independantly for each type
	min_count = 100 # must be at least this many examples to cluster for STR start/end

	pos_STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in pos_class_data['pre_strings']
	])
	pos_STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in pos_class_data['post_strings']
	])
	neg_STR_starts = np.array([
		s[-str_pad_size : -str_pad_size+str_motif_len] for s in neg_class_data['pre_strings']
	])
	neg_STR_ends = np.array([
		s[str_pad_size-str_motif_len : str_pad_size] for s in neg_class_data['post_strings']
	])

	print("Positive class data: {} examples".format(len(pos_class_data['sample_data'])))
	print(*zip(*np.unique(pos_STR_starts, return_counts=True)))
	print(*zip(*np.unique(pos_STR_ends, return_counts=True)))

	print("Negative class data: {} examples".format(len(neg_class_data['sample_data'])))
	print(*zip(*np.unique(neg_STR_starts, return_counts=True)))
	print(*zip(*np.unique(neg_STR_ends, return_counts=True)))

	pos_starts_to_cluster = {
		s for s,c in zip(*np.unique(pos_STR_starts, return_counts=True)) if c >= min_count
	}
	neg_starts_to_cluster = {
		s for s,c in zip(*np.unique(neg_STR_starts, return_counts=True)) if c >= min_count
	}
	starts_to_cluster = pos_starts_to_cluster & neg_starts_to_cluster
	# ends_to_cluster = [
	# 	s for s,c in zip(*np.unique(STR_ends, return_counts=True)) if c >= min_count
	# ]

	# For each STR start pattern subset, cluster seqs by attribution weights
	if cluster_method == 'dbscan':
		clusterer = DBSCAN(
			eps=dbscan_params['eps'],
			min_samples=dbscan_params['min_samples'],
			metric=cluster_metric,
			n_jobs=-1
		)
	elif cluster_method == 'hdbscan':# and not use_weights:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=hdbscan_params['min_cluster_size'],
			min_samples=hdbscan_params['min_samples'],
			cluster_selection_epsilon=hdbscan_params['cluster_selection_epsilon'],
			alpha=hdbscan_params['alpha'],
			metric=cluster_metric,
			core_dist_n_jobs=-1
		)
	else:
		raise ValueError("Invalid cluster_method: {}".format(cluster_method))

	# Create dir to save cluster results
	if cluster_method == 'dbscan':
		cluster_details_str = '_'.join(
			str(v) for v in [dbscan_params['eps'], dbscan_params['min_samples']]
		)
	elif cluster_method == 'hdbscan':
		cluster_details_str = '_'.join([
			str(hdbscan_params['min_cluster_size']),
			str(hdbscan_params['min_samples']),
			str(hdbscan_params['cluster_selection_epsilon']),
			str(hdbscan_params['alpha'])
		])

	if save_plots:
		cluster_res_dir = '_'.join([
			'TPTN' if use_TP_TN else 'label',
			cluster_method,
			cluster_metric,
			cluster_details_str,
			str(n_per_side),
			'clip{}-{}'.format(*clip_thresh) if clip_attrs else 'no_clip',
		])
		cluster_res_dir = os.path.join(
			'cba_2_plots', label_version, model_version, cluster_res_dir
		)

		if os.path.exists(cluster_res_dir):
			i = 1
			while os.path.exists(cluster_res_dir + '_' + str(i)):
				i += 1

			cluster_res_dir = cluster_res_dir + '_' + str(i)
		
		os.makedirs(cluster_res_dir)

	for start_pattern in sorted(starts_to_cluster):
		print(start_pattern)

		# subset data
		pos_subset_data = subset_attr_data(
			pos_class_data,
			pos_STR_starts == start_pattern
		)
		neg_subset_data = subset_attr_data(
			neg_class_data,
			neg_STR_starts == start_pattern
		)

		# get data in useful format and cluster
		X_pos = pos_subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		X_pos = X_pos.reshape(X_pos.shape[0], -1)

		if clip_attrs:
			cutoffs = np.percentile(X_pos, [clip_thresh[0], clip_thresh[1]])
			X_pos = np.clip(X_pos, cutoffs[0], cutoffs[1])

		pos_clust_labels = clusterer.fit_predict(X_pos)
		print("Positive class: {} clusters".format(len(np.unique(pos_clust_labels))))
		print(*zip(*np.unique(pos_clust_labels, return_counts=True)))

		X_neg = neg_subset_data['pre'][:,:4:,-n_per_side-str_pad_size:-str_pad_size]
		X_neg = X_neg.reshape(X_neg.shape[0], -1)

		if clip_attrs:
			cutoffs = np.percentile(X_neg, [clip_thresh[0], clip_thresh[1]])
			X_neg = np.clip(X_neg, cutoffs[0], cutoffs[1])

		neg_clust_labels = clusterer.fit_predict(X_neg)
		print("Negative class: {} clusters".format(len(np.unique(neg_clust_labels))))
		print(*zip(*np.unique(neg_clust_labels, return_counts=True)))

		# Plot clusters for both classes
		pos_fig, _, pos_sorted_labels = plot_cluster_attn_and_ppm(
			pos_clust_labels, 
			pos_subset_data, 
			n_per_side,
			str_pad_size,
			label_desc='TP' if use_TP_TN else 'heterozygous',
			plt_height=.8
		)
		neg_fig, _, neg_sorted_labels = plot_cluster_attn_and_ppm(
			neg_clust_labels, 
			neg_subset_data, 
			n_per_side,
			str_pad_size,
			label_desc='TN' if use_TP_TN else 'non-heterozygous',
			plt_height=.8
		)

		# plot prediction confidence dists and attrs by cluster and class
		pos_conf_df = pd.DataFrame({
			'pred_conf': pos_subset_data['predictions'],
			'cluster': pos_clust_labels,
			'heterozygosity_score': pos_subset_data['sample_data'].heterozygosity.values
		})
	
		pos_conf_attr_fig, axs = plt.subplots(1, 3, 
			figsize=(6.4, pos_fig.get_figheight())
		)
		sns.violinplot(
			x='pred_conf',
			y='cluster',
			orient='h',
			# scale='count',
			order=pos_sorted_labels,
			cut=0,
			data=pos_conf_df,
			ax=axs[0]
		)
		axs[0].set_title("Predicted Likelihood")
		axs[0].set_xlabel(None)
		sns.violinplot(
			x='heterozygosity_score',
			y='cluster',
			orient='h',
			order=pos_sorted_labels,
			cut=0,
			data=pos_conf_df,
			ax=axs[1]
		)
		axs[1].set_ylabel(None)
		axs[1].set_title('Heterozygosity')
		axs[1].set_xlabel(None)
		sns.countplot(
			y='cluster',
			data=pos_conf_df,
			order=pos_sorted_labels,
			ax=axs[2]
		)
		axs[2].set_title('Cluster size')
		axs[2].set_xlabel(None)
		axs[2].set_ylabel(None)

		abs_values = pos_conf_df['cluster'].value_counts(ascending=False)
		rel_values = pos_conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		count_labels = [
			f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in pos_sorted_labels
		]
		axs[2].bar_label(container=axs[2].containers[0], labels=count_labels)
		plt.tight_layout(w_pad=.21)

		# Negative equivalent plot (plot 4/4)
		neg_conf_df = pd.DataFrame({
			'pred_conf': neg_subset_data['predictions'],
			'cluster': neg_clust_labels,
			'heterozygosity_score': neg_subset_data['sample_data'].heterozygosity.values
		})
	
		neg_conf_attr_fig, axs = plt.subplots(1, 2, 
			figsize=(6.4, neg_fig.get_figheight())
		)
		sns.violinplot(
			x='pred_conf',
			y='cluster',
			orient='h',
			# scale='count',
			order=neg_sorted_labels,
			cut=0,
			data=neg_conf_df,
			ax=axs[0]
		)
		# sns.violinplot(
		# 	x='heterozygosity_score',
		# 	y='cluster',
		# 	orient='h',
		# 	order=neg_sorted_labels,
		# 	cut=0,
		# 	data=neg_conf_df,
		# 	ax=axs[1]
		# )
		sns.countplot(
			y='cluster',
			data=neg_conf_df,
			order=neg_sorted_labels,
			ax=axs[1]
		)
		# axs[1].set_xscale('log')
		axs[1].set_xlabel(None)
		axs[1].set_ylabel(None)
		# plt.tight_layout()

		abs_values = neg_conf_df['cluster'].value_counts(ascending=False)
		rel_values = neg_conf_df['cluster'].value_counts(ascending=False, normalize=True) * 100
		count_labels = [
			f'{abs_values[l]} ({rel_values[l]:.1f}%)' for l in neg_sorted_labels
		]
		axs[1].bar_label(container=axs[1].containers[0], labels=count_labels)

		if save_plots:
			pos_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_pos_by_base.png'.format(start_pattern)
			))
			neg_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_neg_by_base.png'.format(start_pattern)
			))
			pos_conf_attr_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_pos_by_cluster.png'.format(start_pattern)
			))
			neg_conf_attr_fig.savefig(os.path.join(
				cluster_res_dir,
				'pre_{}_neg_by_cluster.png'.format(start_pattern)
			))
		if show_plots:
			plt.show()
		
		plt.close()