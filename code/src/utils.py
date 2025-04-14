import numpy as np
import scipy.stats as sps

def norm_pdf(x, mu, sig):
	return sps.norm.pdf(x, mu, sig)

def norm_cdf(x, mu, sig):
	return sps.norm.cdf(x, mu, sig)

def conf_iterval(l, rgs):
	alpha = (100. - rgs)/2.
	low = np.percentile(l, alpha, interpolation='nearest', axis=0)
	high = np.percentile(l, rgs+alpha, interpolation='nearest', axis=0)
	return (low, high)

def remove_empty(l):
	""" recursively remove empty array from nested array
	:param l: (list) nested list with empty list(s)
	:return: (list)
	"""
	return list(filter(lambda x: not isinstance(x, (str, list, list)) or x, (remove_empty(x) if isinstance(x, (list, list)) else x for x in l)))
