import os, functools, time, datetime, copy, tqdm
from decimal import Decimal
import openpyxl
import numpy as np
import pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import multiprocessing as mp
import lmfit as lmf
rng = np.random.RandomState(seed=839212363)
options = {'strings_to_formulas': False, 'strings_to_urls': False}  # for saving fit results to excel file

from src.parse import parse_data
from src.utils import conf_iterval, norm_cdf, norm_pdf
from src.model import Cyton2Model

rc = {
	'figure.figsize': (9, 7),
	# 'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 12,
	# 'xtick.labelsize': 14, 'ytick.labelsize': 14,
	# 'axes.grid': True, 'axes.grid.axis': 'x', 'axes.grid.axis': 'y',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 7.5, 'lines.linewidth': 1,
	'errorbar.capsize': 2.5
}
sns.set(style='white', rc=rc)

DT = 0.1              # [Cyton Model] Time step
TIME_THRESHOLD = 10   # select t > TIME_THRESHOLD data points
ITER_SEARCH = 200     # [Cyton Model] Number of initial search
MAX_NFEV = None       # [LMFIT] Maximum number of function evaluation
LM_FIT_KWS = {        # [LMFIT/SciPy] Key-word arguements pass to LMFIT minimizer for Levenberg-Marquardt algorithm
	# 'ftol': 1E-10,  # Relative error desired in the sum of squares. DEFAULT: 1.49012E-8
	# 'xtol': 1E-10,  # Relative error desired in the approximate solution. DEFAULT: 1.49012E-8
	# 'gtol': 0.0,    # Orthogonality desired between the function vector and the columns of the Jacobian. DEFAULT: 0.0
	'epsfcn': 1E-4    # A variable used in determining a suitable step length for the forward-difference approximation of the Jacobian (for Dfun=None). Normally the actual step length will be sqrt(epsfcn)*x If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision. Default value is around 2E-16. As it turns out, the optimisation routine starts by making a very small move (functinal evaluation) and calculating the finite-difference Jacobian matrix to determine the direction to move. The default value is too small to detect sensitivity of 'b' parameter.
}
RGS = 95              # [BOOTSTRAP] alpha for confidence interval
ITER_BOOTS = 100      # [BOOTSTRAP] Bootstrap samples

### MODELS: RESIDUALS
# MARK: Reprog. death
# previously known as "stimulated" death
def reprogrammed_death(pars, x, data=None):
	"""
	The model fits to t > TIME_THRESHOLD total cohorts
	"""
	vals = pars.valuesdict()
	c1, mDie, sDie = vals['c1'], vals['mDie'], vals['sDie']

	model = c1 * (1. - norm_cdf(x, mDie, sDie))

	return (data - model)

# MARK: Resid
def residual(pars, x, data=None, model=None):
	vals = pars.valuesdict()
	mDiv0, sDiv0 = vals['mDiv0'], vals['sDiv0']
	mDD, sDD = vals['mDD'], vals['sDD']
	mDie, sDie = vals['mDie'], vals['sDie']
	b = vals['b']

	pred = model.evaluate(mDiv0, sDiv0, mDD, sDD, mDie, sDie, b)

	return (data - pred)

# MARK: Bootstrap
def bootstrap(pos, key, df, cond, icnd, hts, nreps, a1_params, a2_params, paramExcl):
	# define set collectors
	boots = {
		'mDiv0': [], 'sDiv0': [],
		'mDD': [], 'sDD': [],
		'mDie': [], 'sDie': [],
		'b': [], 'N0': [], 'N1': [], 'c1': [], '1-pl': []
	}
	for _ in tqdm.trange(ITER_BOOTS, desc="[BOOTSTRAP]", leave=False, position=2*pos+1):
		a1_pars = copy.copy(a1_params)
		a2_pars = copy.copy(a2_params)

		# SHUFFLE ENTIRE DATASET
		#  - I'm only intereseted in t > TIME_THRESHOLD, but the activation fraction depends on the empirical average initial cell numbers. Thus, I must shuffle them all to reflect the bootstrap changes on (1-pl) estimation
		cells = df['cgens']['rep'][icnd]  # n(g,t): number of cells per gen at t
		times_boot, cohorts_boot, avg_init_cohort = [], [], 0  # times & C(t); total cohort numbers
		gens_boot, cells_boot, avg_init_cells = [], [], 0 # generations & raveled n(g,t)
		for itpt, (data, ht) in enumerate(zip(cells, hts)):
			for _ in range(len(data)):
				rand_idx = rng.randint(0, len(data))
				cohorts = 0.
				if ht > TIME_THRESHOLD:
					for igen, cell_number_rep in enumerate(data[rand_idx]):
						cohorts += cell_number_rep * np.power(2., -float(igen))
						gens_boot.append(igen)
						cells_boot.append(cell_number_rep)
					times_boot.append(ht)
					cohorts_boot.append(cohorts)

				if not itpt:
					for igen, cell_number_rep in enumerate(data[rand_idx]):
						avg_init_cohort += cell_number_rep * np.power(2., -float(igen))
						avg_init_cells += cell_number_rep
			if not itpt:
				avg_init_cohort = avg_init_cohort/float(len(data))
				avg_init_cells = avg_init_cells/float(len(data))

		times_boot = np.asarray(times_boot)
		cohorts_boot = np.asarray(cohorts_boot)
		gens_boot = np.asarray(gens_boot)
		cells_boot = np.asarray(cells_boot)

		# a1_pars['c1'].set(max=np.max(cohorts_boot))

		# Initiate search on the bootstrap sample
		a1_boots_candidates = {'result': [], 'residual': []}
		for _ in tqdm.trange(ITER_SEARCH, desc=f"[Death] > {key} > {cond}", leave=False, position=2*pos+2):
			for par in a1_pars:
				if a1_pars[par].vary:
					a1_par_min, a1_par_max = a1_pars[par].min, a1_pars[par].max
					a1_pars[par].set(value=rng.uniform(low=a1_par_min, high=a1_par_max))
			try:
				mini = lmf.Minimizer(reprogrammed_death, a1_pars, fcn_args=(times_boot, cohorts_boot), **LM_FIT_KWS)
				res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)

				a1_boots_candidates['result'].append(res)
				a1_boots_candidates['residual'].append(res.chisqr)
			except ValueError as ve:
				pass
		# extract parameters
		a1_boots_fit_results = pd.DataFrame(a1_boots_candidates)
		a1_boots_fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

		a1_boots_vals = a1_boots_fit_results.iloc[0]['result'].params.valuesdict()
		c1, a1_mDie, a1_sDie = a1_boots_vals['c1'], a1_boots_vals['mDie'], a1_boots_vals['sDie']
		a2_pars['mDie'].set(value=a1_mDie)
		a2_pars['sDie'].set(value=a1_sDie)

		total_cohorts_at_0h = c1 * (1. - norm_cdf(0.0, a1_mDie, a1_sDie))  # find total cohorts at 0h
		oneMinusPL = total_cohorts_at_0h/avg_init_cohort

		# redefine the cyton model
		cond_hts = np.asarray(hts)
		prime_hts = cond_hts[cond_hts > TIME_THRESHOLD]
		# prime_idx = np.argwhere(cond_hts > TIME_THRESHOLD)[0][0]
		N0 = avg_init_cells
		N1 = N0 * oneMinusPL

		model = Cyton2Model(prime_hts, N1, int(max(gens_boot)), DT, nreps, logn=False)  # define cyton model object

		candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
		for _ in tqdm.trange(ITER_SEARCH, desc=f"[Cyton] > {key} > {cond}", leave=False, position=2*pos+2):
			# Random initial values
			for par in a2_pars:
				if par in paramExcl: pass  # Ignore excluded parameters
				else:
					par_min, par_max = a2_pars[par].min, a2_pars[par].max  # determine its min and max range
					a2_pars[par].set(value=rng.uniform(low=par_min, high=par_max))

			try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
				mini = lmf.Minimizer(residual, a2_pars, fcn_args=(gens_boot, cells_boot, model), **LM_FIT_KWS)
				res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm

				candidates['result'].append(res)
				candidates['residual'].append(res.chisqr)
			except ValueError as ve:
				pass

		fit_results = pd.DataFrame(candidates)
		fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

		boot_best_fit = fit_results.iloc[0]['result'].params.valuesdict()
		for var in boot_best_fit:
			boots[var].append(boot_best_fit[var])
		boots['N0'].append(N0)
		boots['N1'].append(N1)
		boots['c1'].append(c1)
		boots['1-pl'].append(oneMinusPL)
	boots = pd.DataFrame(boots)
	return boots

# MARK: Fit
def fit(inputs):
	key, df, reader, icnd, pos = inputs

	hts = reader.harvested_times[icnd]
	mgen = reader.generation_per_condition[icnd]
	condition = reader.condition_names[icnd]

	### PREPARE DATA
	data = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
	# Manually ravel the data. This allows asymmetric replicate numbers.
	x_times, y_cohorts = [], []  # times & C(t); total cohort numbers
	x_gens, y_cells = [], [] # generations & raveled n(g,t)
	all_reps, nreps = [len(l) for l in data], []  # get number of replicates in a sample

	for datum, ht in zip(data, hts):
		if ht > TIME_THRESHOLD:
			for irep, rep in enumerate(datum):
				cohorts = 0.
				for igen, cell_number in enumerate(rep):
					cohorts += cell_number * np.power(2., -float(igen))
					x_gens.append(igen)
					y_cells.append(cell_number)
				x_times.append(ht)
				y_cohorts.append(cohorts)
			nreps.append(irep+1)

	x_times = np.asarray(x_times)
	y_cohorts = np.asarray(y_cohorts)
	x_gens = np.asarray(x_gens)
	y_cells = np.asarray(y_cells)

	#################################################################
	# ANALYSIS 1: ESTIMATE REPORGRAMMED DEATH & ACTIVATION FRACTION #
	#################################################################
	# define parameters for analysis 1
	_tmp_cohort_rep = []
	for cohort_rep in df['cohorts_gens']['rep'][icnd]:
		for _rep in cohort_rep:
			_tmp_cohort_rep.append(np.sum(_rep))

	a1_params = lmf.Parameters()
	a1_params.add('c1', value=3000, min=0, max=max(_tmp_cohort_rep), vary=True)

	a1_params.add('mDie', value=60, min=0.001, max=300, vary=True)
	a1_params.add('sDie', value=10, min=0.001, max=100, vary=True)

	# initiate search to find the best optimal parameters
	a1_candidates = {'result': [], 'residual': []}
	for _ in tqdm.trange(ITER_SEARCH, desc=f"[A1 SEARCH] > {key} > {condition}", leave=False, position=2*pos+1):
		# draw random initial guesses from uniform distribution within (min, max) ranges defined in "a1_params"
		for par in a1_params:
			if a1_params[par].vary:
				par_min, par_max = a1_params[par].min, a1_params[par].max
				a1_params[par].set(value=rng.uniform(low=par_min, high=par_max))
		try:
			mini = lmf.Minimizer(reprogrammed_death, a1_params, fcn_args=(x_times, y_cohorts), **LM_FIT_KWS)
			res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)

			a1_candidates['result'].append(res)
			a1_candidates['residual'].append(res.chisqr)
		except ValueError as ve:
			pass
	# extract parameters
	a1_fit_results = pd.DataFrame(a1_candidates)
	a1_fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

	a1_vals = a1_fit_results.iloc[0]['result'].params.valuesdict()
	c1, a1_mDie, a1_sDie = a1_vals['c1'], a1_vals['mDie'], a1_vals['sDie']

	total_cohorts_at_0h = c1 * (1. - norm_cdf(0.0, a1_mDie, a1_sDie))  # find total cohorts at 0h

	# compute activation fraction
	c0 = np.sum(df['cohorts_gens']['avg'][icnd][0])  # empirical average cohorts at t=0
	oneMinusPL = total_cohorts_at_0h/c0

	############################################################
	# ANALYSIS 2: ESTIMATE DIVISION PARAMETERS VIA CYTON MODEL #
	############################################################
	cond_hts = np.asarray(hts)
	prime_hts = cond_hts[cond_hts > TIME_THRESHOLD]
	prime_idx = np.argwhere(cond_hts > TIME_THRESHOLD)[0][0]

	if key == 'DV24_001 CsA MPA OTI BimKO':
		pars = {  # Initial values
			'mDiv0': 30, 'sDiv0': 0.2,         # Time to first division
			'mDD': 65.32, 'sDD': 14.583,       # Time to division destiny (Fitted from No Drug)
			'mDie': a1_mDie, 'sDie': a1_sDie,  # Time to death
			'b': 10							   # Subseqeunt division time
		}
	elif key == 'DV23_004 DEX+CsA':
		pars = {  # Initial values
			'mDiv0': 30, 'sDiv0': 0.2,
			'mDD': 58.72, 'sDD': 12.167,       # Fitted from No Drug
			'mDie': a1_mDie, 'sDie': a1_sDie,
			'b': 10
		}
	bounds = {
		'lb': {  # Lower bounds
			'mDiv0': 1E-2, 'sDiv0': 1E-2,
			'mDD': 1E-2, 'sDD': 1E-2,
			'mDie': 1E-2, 'sDie': 1E-2,
			'b': 5
		},
		'ub': {  # Upper bounds
			'mDiv0': 100, 'sDiv0': 50,
			'mDD': 150, 'sDD': 100,
			'mDie': 300, 'sDie': 100,
			'b': 50
		}
	}
	vary = {  # True = Subject to change; False = Lock parameter
		'mDiv0': True, 'sDiv0': True,
		'mDD': False, 'sDD': False,
		'mDie': False, 'sDie': False,
		'b': True
	}
	
	a2_params = lmf.Parameters()
	# LMFIT add parameter properties with tuples: (NAME, VALUE, VARY, MIN, MAX, EXPR, BRUTE_STEP)
	for par in pars:
		a2_params.add(par, value=pars[par], min=bounds['lb'][par], max=bounds['ub'][par], vary=vary[par])
	paramExcl = [p for p in a2_params if not a2_params[p].vary]  # List of parameters excluded from fitting (i.e. vary=False)

	N0 = df['cells']['avg'][icnd][0]
	N1 = N0 * oneMinusPL  # Initial cell number = # cells measured at the first time point
	model = Cyton2Model(prime_hts, N1, mgen, DT, nreps, logn=False)

	candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
	for _ in tqdm.trange(ITER_SEARCH, desc=f"[A2 SEARCH] > {key} > {condition}", leave=False, position=2*pos+1):
		# Random initial values
		for par in a2_params:
			if par in paramExcl: pass  # Ignore excluded parameters
			else:
				par_min, par_max = a2_params[par].min, a2_params[par].max  # determine its min and max range
				a2_params[par].set(value=rng.uniform(low=par_min, high=par_max))

		try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
			mini = lmf.Minimizer(residual, a2_params, fcn_args=(x_gens, y_cells, model), **LM_FIT_KWS)
			res = mini.minimize(method='leastsq', max_nfev=MAX_NFEV)  # Levenberg-Marquardt algorithm

			candidates['result'].append(res)
			candidates['residual'].append(res.chisqr)
		except ValueError as ve:
			pass
	fit_results = pd.DataFrame(candidates)
	fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

	# Extract best-fit parameters
	best_fit = fit_results.iloc[0]['result'].params.valuesdict()
	mDiv0, sDiv0 = best_fit['mDiv0'], best_fit['sDiv0']
	mDD, sDD = best_fit['mDD'], best_fit['sDD']
	mDie, sDie = best_fit['mDie'], best_fit['sDie']
	b = best_fit['b']

	### RUN BOOTSTRAP
	boots = bootstrap(pos, key, df, condition, icnd, hts, nreps, a1_params, a2_params, paramExcl)

	### PLOT RESULTS
	t0, tf = 0, 150
	times = np.linspace(t0, tf, num=int(tf/DT)+1)
	gens = np.array([i for i in range(mgen+1)])

	# Calculate bootstrap intervals
	tdiv0_pdf_curves, tdiv0_cdf_curves = [], []
	tdd_pdf_curves, tdd_cdf_curves = [], []
	tdie_pdf_curves, tdie_cdf_curves = [], []

	b_ext_total_live_cells, b_ext_cells_per_gen = [], []
	b_hts_total_live_cells, b_hts_cells_per_gen = [], []

	conf = {
		'tdiv0_pdf': [], 'tdiv0_cdf': [], 'tdd_pdf': [], 'tdd_cdf': [], 'tdie_pdf': [], 'tdie_cdf': [],
		'ext_total_cohorts': [], 'ext_total_live_cells': [], 'ext_cells_per_gen': [], 'hts_total_live_cells': [], 'hts_cells_per_gen': [],
		'ext_mdn': [], 'ext_mdn_ex0': []
	}
	tmp_N0, tmp_N1, tmp_c1, tmp_oneMinusPL = [], [], [], []  # just to calculate confidence interval for N0, but this is not real parameter! The interval is only from bootstrapping... (And recording this would make easier to plot in the future)
	for bsample in boots.iterrows():
		b_mDiv0, b_sDiv0, b_mDD, b_sDD, b_mDie, b_sDie, b_b, b_N0, b_N1, b_c1, b_oneMinusPL = bsample[1].values
		b_params = a2_params.copy()
		b_params['mDiv0'].set(value=b_mDiv0); b_params['sDiv0'].set(value=b_sDiv0)
		b_params['mDD'].set(value=b_mDD); b_params['sDD'].set(value=b_sDD)
		b_params['mDie'].set(value=b_mDie); b_params['sDie'].set(value=b_sDie)
		b_params['b'].set(value=b_b)

		# Calculate PDF and CDF curves for each set of parameter
		b_tdiv0_pdf, b_tdiv0_cdf = norm_pdf(times, b_mDiv0, b_sDiv0), norm_cdf(times, b_mDiv0, b_sDiv0)
		b_tdd_pdf, b_tdd_cdf = norm_pdf(times, b_mDD, b_sDD), norm_cdf(times, b_mDD, b_sDD)
		b_tdie_pdf, b_tdie_cdf = norm_pdf(times, b_mDie, b_sDie), norm_cdf(times, b_mDie, b_sDie)

		tdiv0_pdf_curves.append(b_tdiv0_pdf); tdiv0_cdf_curves.append(b_tdiv0_cdf)
		tdd_pdf_curves.append(b_tdd_pdf); tdd_cdf_curves.append(b_tdd_cdf)
		tdie_pdf_curves.append(b_tdie_pdf); tdie_cdf_curves.append(b_tdie_cdf)

		# Calculate model prediction for each set of parameter
		b_model = Cyton2Model(hts, b_N1, mgen, DT, nreps, logn=False)
		b_extrapolate = b_model.extrapolate(times, b_params)  # get extrapolation for all "times" (discretised) and at harvested timepoints
		b_ext_total_live_cells.append(b_extrapolate['ext']['total_live_cells'])
		b_ext_total_cohorts = np.sum(np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens), axis=1)
		b_ext_cells_per_gen.append(b_extrapolate['ext']['cells_gen'])
		b_hts_total_live_cells.append(b_extrapolate['hts']['total_live_cells'])
		b_hts_cells_per_gen.append(b_extrapolate['hts']['cells_gen'])

		_b_cohorts = np.transpose(b_extrapolate['ext']['cells_gen']) * np.power(2.,-gens)  # compute MDN predictions
		_b_weighted = _b_cohorts * gens
		b_ext_mdn = np.sum(_b_weighted, axis=1) / b_ext_total_cohorts

		_b_cohorts_ex0 = np.transpose(b_extrapolate['ext']['cells_gen'][1:]) * np.power(2.,-gens[1:])  # compute MDN predictions excluding gen0
		_b_weighted_ext0 = _b_cohorts_ex0 * gens[1:]
		b_ext_mdn_ex0 = np.sum(_b_weighted_ext0, axis=1) / np.sum(np.transpose(b_extrapolate['ext']['cells_gen'][1:]) * np.power(2.,-gens[1:]), axis=1)

		conf['tdiv0_pdf'].append(b_tdiv0_pdf); conf['tdiv0_cdf'].append(b_tdiv0_cdf)
		conf['tdd_pdf'].append(b_tdd_pdf); conf['tdd_cdf'].append(b_tdd_cdf)
		conf['tdie_pdf'].append(b_tdie_pdf); conf['tdie_cdf'].append(b_tdie_cdf)
		conf['ext_total_cohorts'].append(b_ext_total_cohorts)
		conf['ext_total_live_cells'].append(b_ext_total_live_cells); conf['ext_cells_per_gen'].append(b_ext_cells_per_gen)
		conf['hts_total_live_cells'].append(b_hts_total_live_cells); conf['hts_cells_per_gen'].append(b_hts_cells_per_gen)
		conf['ext_mdn'].append(b_ext_mdn)
		conf['ext_mdn_ex0'].append(b_ext_mdn_ex0)

		tmp_N0.append(b_N0)
		tmp_N1.append(b_N1)
		tmp_c1.append(b_c1)
		tmp_oneMinusPL.append(b_oneMinusPL)

	# Calculate 95% confidence bands on PDF, CDF and model predictions
	for obj in conf:
		stack = np.vstack(conf[obj])
		conf[obj] = conf_iterval(stack, RGS)

	# 95% confidence interval on each parameter values
	err_mDiv0, err_sDiv0 = conf_iterval(boots['mDiv0'], RGS), conf_iterval(boots['sDiv0'], RGS)
	err_mDD, err_sDD = conf_iterval(boots['mDD'], RGS), conf_iterval(boots['sDD'], RGS)
	err_mDie, err_sDie = conf_iterval(boots['mDie'], RGS), conf_iterval(boots['sDie'], RGS)
	err_b = conf_iterval(boots['b'], RGS)

	err_N0 = conf_iterval(tmp_N0, RGS)
	err_N1 = conf_iterval(tmp_N1, RGS)  # AGAIN, NOT A REAL PARAMETER
	err_c1 = conf_iterval(tmp_c1, RGS)
	err_oneMinusPL = conf_iterval(tmp_oneMinusPL, RGS)

	save_best_fit = pd.DataFrame(
		data={"best-fit": [mDiv0, sDiv0, mDD, sDD, mDie, sDie, b, N0, N1, c1, oneMinusPL],
			"low95": [mDiv0-err_mDiv0[0], sDiv0-err_sDiv0[0], mDD-err_mDD[0], sDD-err_sDD[0], mDie-err_mDie[0], sDie-err_sDie[0], b-err_b[0], N0-err_N0[0], N1-err_N1[0], c1-err_c1[0], oneMinusPL-err_oneMinusPL[0]],
			"high95": [err_mDiv0[1]-mDiv0, err_sDiv0[1]-sDiv0, err_mDD[1]-mDD, err_sDD[1]-sDD, err_mDie[1]-mDie, err_sDie[1]-sDie, err_b[1]-b, err_N0[1]-N0, err_N1[1]-N1, err_c1[1]-c1, err_oneMinusPL[1]-oneMinusPL],
			"vary": np.append([a2_params[p].vary for p in a2_params], ["False", "False", "False", "False"])}, 
		index=["mDiv0", "sDiv0", "mDD", "sDD", "mDie", "sDie", "b", "N0", "N1", "c1", "1-pl"]) 
	
	excel_path = f"./out/{key}_{condition}_result.xlsx"
	if os.path.isfile(excel_path):
		with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
			save_best_fit.to_excel(writer, sheet_name=f"pars_{condition}")
			boots.to_excel(writer, sheet_name=f"boot_{condition}")
	else:
		with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
			save_best_fit.to_excel(writer, sheet_name=f"pars_{condition}")
			boots.to_excel(writer, sheet_name=f"boot_{condition}")

	# Get extrapolation: 1. for given time range t \in [t0, tf]; 2. at harvested time points
	model = Cyton2Model(hts, N1, mgen, DT, nreps, logn=False)
	extrapolate = model.extrapolate(times, best_fit)  # get extrapolation for all "times" (discretised) and at harvested timepoints
	ext_total_live_cells = extrapolate['ext']['total_live_cells']
	ext_cells_per_gen = extrapolate['ext']['cells_gen']
	hts_total_live_cells = extrapolate['hts']['total_live_cells']
	hts_cells_per_gen = extrapolate['hts']['cells_gen']

	# Calculate PDF and CDF
	tdiv0_pdf, tdiv0_cdf = norm_pdf(times, mDiv0, sDiv0), norm_cdf(times, mDiv0, sDiv0)
	tdd_pdf, tdd_cdf = norm_pdf(times, mDD, sDD), norm_cdf(times, mDD, sDD)
	tdie_pdf, tdie_cdf = norm_pdf(times, mDie, sDie), norm_cdf(times, mDie, sDie)
	### FIG 1: SUMMARY PLOT
	fig1, ax1 = plt.subplots(nrows=2, ncols=2, sharex=True)
	fig1.suptitle(f"[{key}][{condition}] Summary")

	## MARK: MDN
	ax1[0,0].set_ylabel("MDN")
	ax1[0,0].set_xlabel("Time (hour)")
	mdn_tps = []
	mdns = []
	mdns_ex0 = []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			mdn_tps.append(ht)
			_cgens = np.array(df['cgens']['rep'][icnd][itpt][irep])
			_cohorts = _cgens / np.power(2., gens)
			weighted = _cohorts * gens
			_mdn = np.sum(weighted) / np.sum(_cohorts)
			mdns.append(_mdn)

			_cohorts_ex0 = _cgens[1:] / np.power(2., gens[1:])
			weighted_ex0 = _cohorts_ex0 * gens[1:]
			_mdn_ex0 = np.sum(weighted_ex0) / np.sum(_cohorts_ex0)
			mdns_ex0.append(_mdn_ex0)
	ax1[0,0].plot(mdn_tps, mdns, 'r.', label='data')
	ax1[0,0].plot(mdn_tps, mdns_ex0, '.', color='navy', mfc='none', label='data (ex. gen0)')
	_ext_cohorts = np.transpose(ext_cells_per_gen) * np.power(2.,-gens)
	_ext_weighted = _ext_cohorts * gens
	ext_mdn = np.sum(_ext_weighted, axis=1) / np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
	ax1[0,0].plot(times, ext_mdn, 'k-', label='model')
	ax1[0,0].fill_between(times, conf['ext_mdn'][0], conf['ext_mdn'][1], fc='k', ec=None, alpha=0.3)
	## MDN excluding gen.0
	_ext_cohorts_ex0 = np.transpose(ext_cells_per_gen[1:]) * np.power(2.,-gens[1:])
	_ext_weighted_ext0 = _ext_cohorts_ex0 * gens[1:]
	ext_mdn_ex0 = np.sum(_ext_weighted_ext0, axis=1) / np.sum(np.transpose(ext_cells_per_gen[1:]) * np.power(2.,-gens[1:]), axis=1)
	ax1[0,0].plot(times, ext_mdn_ex0, '--', color='navy', label='model (ex. gen0)')
	ax1[0,0].fill_between(times, conf['ext_mdn_ex0'][0], conf['ext_mdn_ex0'][1], fc='navy', ec=None, alpha=0.3)
	ax1[0,0].set_ylim(bottom=0)
	ax1[0,0].legend(fontsize=9, frameon=True)

	## MARK: Total cohorts
	ax1[0,1].set_title(f"$1-pl = {oneMinusPL:.4f} \pm_{{{oneMinusPL-err_oneMinusPL[0]:.4f}}}^{{{err_oneMinusPL[1]-oneMinusPL:.4f}}}$")
	ax1[0,1].set_ylabel("Cohort number")
	tps_excl, total_cohorts_excl = [], []
	tps_incl, total_cohorts_incl = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			if ht > TIME_THRESHOLD:
				tps_incl.append(ht)
				total_cohorts_incl.append(np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep]))
			else:
				tps_excl.append(ht)
				total_cohorts_excl.append(np.sum(df['cohorts_gens']['rep'][icnd][itpt][irep]))
	ext_total_cohorts = np.sum(np.transpose(ext_cells_per_gen) * np.power(2.,-gens), axis=1)
	ax1[0,1].plot(tps_excl, total_cohorts_excl, 'bx', label='excluded')
	ax1[0,1].plot(tps_incl, total_cohorts_incl, 'r.', label='data')
	ax1[0,1].plot(times, ext_total_cohorts, 'k-', label='model')
	ax1[0,1].fill_between(times, conf['ext_total_cohorts'][0], conf['ext_total_cohorts'][1], fc='k', ec=None, alpha=0.3)
	ax1[0,1].set_ylim(bottom=0)
	ax1[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[0,1].yaxis.major.formatter._useMathText = True
	ax1[0,1].legend(fontsize=9, frameon=True)

	## MARK: PDFs
	ax1[1,0].set_title(f"$b={b:.2f}h \pm_{{{b-err_b[0]:.2f}}}^{{{err_b[1]-b:.2f}}}$")
	ax1[1,0].set_ylabel("Density")
	label_Tdiv0 = f"$T_{{div}}^0 \sim \mathcal{{N}}({mDiv0:.2f}\pm_{{{mDiv0-err_mDiv0[0]:.2f}}}^{{{err_mDiv0[1]-mDiv0:.2f}}}, {sDiv0:.3f} \pm_{{{sDiv0-err_sDiv0[0]:.3f}}}^{{{err_sDiv0[1]-sDiv0:.3f}}})$"
	label_Tdd = f"$T_{{dd}} \sim \mathcal{{N}}({mDD:.2f}\pm_{{{mDD-err_mDD[0]:.2f}}}^{{{err_mDD[1]-mDD:.2f}}}, {sDD:.3f}\pm_{{{sDD-err_sDD[0]:.3f}}}^{{{err_sDD[1]-sDD:.3f}}})$"
	label_Tdie = f"$T_{{die}} \sim \mathcal{{N}}({mDie:.2f}\pm_{{{mDie-err_mDie[0]:.2f}}}^{{{err_mDie[1]-mDie:.2f}}}, {sDie:.3f}\pm_{{{sDie-err_sDie[0]:.3f}}}^{{{err_sDie[1]-sDie:.3f}}})$"
	ax1[1,0].plot(times, tdiv0_pdf, color='blue', ls='-', label=label_Tdiv0)
	ax1[1,0].fill_between(times, conf['tdiv0_pdf'][0], conf['tdiv0_pdf'][1], fc='blue', ec=None, alpha=0.5)
	ax1[1,0].plot(times, tdd_pdf, color='green', ls='-', label=label_Tdd)
	ax1[1,0].fill_between(times, conf['tdd_pdf'][0], conf['tdd_pdf'][1], fc='green', ec=None, alpha=0.5)
	ax1[1,0].plot(times, -tdie_pdf, color='red', ls='-', label=label_Tdie)
	ax1[1,0].fill_between(times, -conf['tdie_pdf'][0], -conf['tdie_pdf'][1], fc='red', ec=None, alpha=0.5)
	ax1[1,0].set_yticks(ax1[1,0].get_yticks())
	ax1[1,0].set_yticklabels(np.round(np.abs(ax1[1,0].get_yticks()), 5))  # remove negative y-tick labels
	ax1[1,0].legend(fontsize=9, frameon=True)

	## MARK: Total cells
	ax1[1,1].set_title(f"$N_0 = {N0:.1f}\pm_{{{N0-err_N0[0]:.1f}}}^{{{err_N0[1]-N0:.1f}}}$, $N_1 = {N1:.1f}\pm_{{{N1-err_N1[0]:.1f}}}^{{{err_N1[1]-N1:.1f}}}$")
	ax1[1,1].set_ylabel("Cell number")
	ax1[1,1].set_xlabel("Time (hour)")
	tps, total_cells = [], []
	for itpt, ht in enumerate(hts):
		for irep in range(all_reps[itpt]):
			if ht > TIME_THRESHOLD:
				tps.append(ht)
				total_cells.append(df['cells']['rep'][icnd][itpt][irep])
	ax1[1,1].plot(tps, total_cells, 'r.')
	ax1[1,1].plot(times, ext_total_live_cells, 'k-', lw=1)
	ax1[1,1].fill_between(times, conf['ext_total_live_cells'][0], conf['ext_total_live_cells'][1], fc='k', ec=None, alpha=0.3)
	cp = sns.hls_palette(mgen+1, l=0.4, s=0.5)
	for igen in range(mgen+1):
		ax1[1,1].errorbar(prime_hts, np.transpose(df['cgens']['avg'][icnd][prime_idx:])[igen], yerr=np.transpose(df['cgens']['sem'][icnd][prime_idx:])[igen], c=cp[igen], fmt='.', ms=5, label=f"Gen {igen}")
		ax1[1,1].plot(times, ext_cells_per_gen[igen], c=cp[igen])
		ax1[1,1].fill_between(times, conf['ext_cells_per_gen'][0][igen], conf['ext_cells_per_gen'][1][igen], fc=cp[igen], ec=None, alpha=0.5)
	ax1[1,1].set_ylim(bottom=0)
	ax1[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax1[1,1].yaxis.major.formatter._useMathText = True
	ax1[1,1].legend(fontsize=9, frameon=True)
	for ax in ax1.flat:
		ax.set_xlim(t0, max(times))
	fig1.tight_layout(rect=(0, 0, 1, 1))

	### FIG 2: CELL NUMBERS PER GENERATION AT HARVESTED TIME POINTS
	if len(hts) <= 6: nrows, ncols = 2, 3
	elif 6 < len(hts) <= 9: nrows, ncols = 3, 3
	else: nrows, ncols = 4, 3

	fig2 = plt.figure()
	fig2.text(0.5, 0.04, "Generations", ha='center', va='center')
	fig2.text(0.02, 0.5, "Cell number", ha='center', va='center', rotation=90)
	axes = []  # store axis
	for itpt, ht in enumerate(hts):
		ax2 = plt.subplot(nrows, ncols, itpt+1)
		ax2.set_axisbelow(True)
		ax2.plot(gens, hts_cells_per_gen[itpt], 'o-', c='k', ms=5, label='model')
		ax2.fill_between(gens, conf['hts_cells_per_gen'][0][itpt], conf['hts_cells_per_gen'][1][itpt], fc='k', ec=None, alpha=0.3)
		for irep in range(all_reps[itpt]):
			if ht > TIME_THRESHOLD:
				ax2.plot(gens, df['cgens']['rep'][icnd][itpt][irep], 'r.', label='data')
			else:
				ax2.plot(gens, df['cgens']['rep'][icnd][itpt][irep], 'bx', label='excluded')
		ax2.set_xticks(gens)
		ax2.annotate(f"{ht}h", xy=(0.75, 0.85), xycoords='axes fraction')
		ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
		ax2.yaxis.major.formatter._useMathText = True
		if itpt not in [len(hts)-3, len(hts)-2, len(hts)-1]:
			ax2.set_xticklabels([])
		if itpt not in [0, 3, 6, 9, 12]:
			ax2.set_yticklabels([])
		ax2.spines['right'].set_visible(True)
		ax2.spines['top'].set_visible(True)
		ax2.grid(True, ls='--')
		axes.append(ax2)
	max_ylim = 0
	for ax in axes:
		_, ymax = ax.get_ylim()
		if max_ylim < ymax:
			max_ylim = ymax
	for ax in axes:
		ax.set_ylim(top=max_ylim)
	handles, labels = [], []
	for ax in fig2.axes:
		hs, ls = ax.get_legend_handles_labels()
		for h, l in zip(hs, ls):
			if l not in labels:
				handles.append(h)
				labels.append(l)
	fig2.legend(handles=handles, labels=labels, ncol=3, bbox_to_anchor=(1,1))
	fig2.tight_layout(rect=(0.02, 0.03, 1, 1))
	fig2.subplots_adjust(wspace=0.05, hspace=0.15)


	### FIG 3: DISTRIBUTION OF BOOTSTRAP SAMPLES 
	alpha = (1. - RGS/100.)/2
	quantiles = boots[['mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'b', '1-pl', 'N0', 'N1']].quantile([alpha, alpha + RGS/100.], numeric_only=True, interpolation='nearest')
	
	xlabs = [r"median ($m_{div}^0$)", r"log-variance ($s_{div}^0$)", 
						r"median ($m_{dd}$)", r"log-variance ($s_{dd}$)", 
						r"median ($m_{die}$)", r"log-variance ($s_{die}$)", 
						r"$b$ (hour)", r"$1-pl$ (Proportion)",
						r"$N_0 (Cells)$", r"$N_1$ (Cells)"]
	colors = ['blue', 'blue', 'green', 'green', 'red', 'red', 'navy', 'k', 'grey', 'grey']
	fig3, ax3 = plt.subplots(nrows=5, ncols=2, figsize=(9, 8))
	fig3.suptitle(f"[{condition}] Bootstrap marginal distribution")
	ax3 = ax3.flat
	for i, obj in enumerate(['mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie', 'b', '1-pl', 'N0', 'N1']):
		if obj in list(best_fit.keys()):
			best = best_fit[obj]
		else:
			if obj=='N0':
				best = N0
			elif obj=='N1':
				best = N1
			else:
				best = boots[obj].mean()
		b_sample = boots[obj].to_numpy()
		l_quant, h_quant = quantiles.iloc[0][obj], quantiles.iloc[1][obj]

		ax3[i].axvline(best, ls='-', c='k', label=f"best-fit={best:.2f}")
		ax3[i].axvline(l_quant, ls=':', c='red', label=f"lo={l_quant:.2f}")
		ax3[i].axvline(h_quant, ls='-.', c='red', label=f"hi={h_quant:.2f}")
		sns.distplot(b_sample, kde=False, hist_kws=dict(ec='k', lw=1), color=colors[i], ax=ax3[i])
		ax3[i].set_xlabel(xlabs[i])
		ax3[i].legend(fontsize=9, loc='upper right')
	fig3.tight_layout(rect=(0, 0, 1, 1))
	fig3.subplots_adjust(wspace=0.1, hspace=0.85)

	# MARK: Save plots
	with PdfPages(f"./out/{key}_{condition}.pdf") as pdf:
		pdf.savefig(fig1)
		pdf.savefig(fig2)
		pdf.savefig(fig3)

	# MARK: Save model outputs
	# Save model ouputs in an Excel file
	dfCohortMethod = pd.DataFrame({
		'Time (hour)': times,
		'Total cohorts': ext_total_cohorts,
		'Total cohorts (low95)': conf['ext_total_cohorts'][0],
		'Total cohorts (upp95)': conf['ext_total_cohorts'][1],
		'MDN': ext_mdn,
		'MDN (low95)': conf['ext_mdn'][0],
		'MDN (upp95)': conf['ext_mdn'][1],
		'MDN.excl.gen0': ext_mdn_ex0,
		'MDN.excl.gen0 (low95)': conf['ext_mdn_ex0'][0],
		'MDN.excl.gen0 (upp95)': conf['ext_mdn_ex0'][1]
	})
	dfPDFs = pd.DataFrame({
		'Time (hour)': times,
		'Tdiv0': tdiv0_pdf,
		'Tdiv0 (low95)': conf['tdiv0_pdf'][0],
		'Tdiv0 (upp95)': conf['tdiv0_pdf'][1],
		'Tdd': tdd_pdf,
		'Tdd (low95)': conf['tdd_pdf'][0],
		'Tdd (upp95)': conf['tdd_pdf'][1],
		'Tdie': tdie_pdf,
		'Tdie (low95)': conf['tdie_pdf'][0],
		'Tdie (upp95)': conf['tdie_pdf'][1]
	})
	dfModelCells = pd.DataFrame({
		'Time (hour)': times,
		'Total cells': ext_total_live_cells,
		'Total cells (low95)': conf['ext_total_live_cells'][0],
		'Total cells (upp95)': conf['ext_total_live_cells'][1]
	})
	for igen in range(mgen+1):
		dfModelCells[f'Gen{igen}'] = ext_cells_per_gen[igen]
		dfModelCells[f'Gen{igen} (low95)'] = conf['ext_cells_per_gen'][0][igen]
		dfModelCells[f'Gen{igen} (upp95)'] = conf['ext_cells_per_gen'][1][igen]
	
	dfModelCellsGen = pd.DataFrame({
		'Gens': gens
	})
	for itpt, ht in enumerate(hts):
		dfModelCellsGen[f'{ht}h'] = hts_cells_per_gen[itpt]
		dfModelCellsGen[f'{ht}h (low95)'] = conf['hts_cells_per_gen'][0][itpt]
		dfModelCellsGen[f'{ht}h (upp95)'] = conf['hts_cells_per_gen'][1][itpt]

	ex_path = f"./out/{key}_{condition}_modelOutputs.xlsx"
	with pd.ExcelWriter(ex_path, engine='openpyxl', mode='w') as writer:
		dfCohortMethod.to_excel(writer, index=False, sheet_name="Cohort Method")
		dfPDFs.to_excel(writer, index=False, sheet_name="PDFs")
		dfModelCells.to_excel(writer, index=False, sheet_name="Cyton2(Cells)")
		dfModelCellsGen.to_excel(writer, index=False, sheet_name="Cyton2(CellsPerGen)")

		auto_adjust_xlsx_column_width(dfCohortMethod, writer, "Cohort Method")
		auto_adjust_xlsx_column_width(dfPDFs, writer, "PDFs")
		auto_adjust_xlsx_column_width(dfModelCells, writer, "Cyton2(Cells)")
		auto_adjust_xlsx_column_width(dfModelCellsGen, writer, "Cyton2(CellsPerGen)")

def auto_adjust_xlsx_column_width(df, writer, sheet_name, margin=0, length_factor=1.0, decimals=3, index=False):
	def text_length(text):
		"""
		Get the effective text length in characters, taking into account newlines
		"""
		if not text:
			return 0
		lines = text.split("\n")
		return max(len(line) for line in lines)

	def _to_str_for_length(v, decimals=3):
		"""
		Like str() but rounds decimals to predefined length
		"""
		if isinstance(v, float):
			# Round to [decimal] places
			return str(Decimal(v).quantize(Decimal('1.' + '0' * decimals)).normalize())
		else:
			return str(v)
	
	sheet = writer.sheets[sheet_name]
	_to_str = functools.partial(_to_str_for_length, decimals=decimals)
	# Compute & set column width for each column
	for column_name in df.columns:
		# Convert the value of the columns to string and select the 
		column_length = max(df[column_name].apply(_to_str).map(text_length).max(), text_length(column_name)) + 5
		# Get index of column in XLSX
		# Column index is +1 if we also export the index column
		col_idx = df.columns.get_loc(column_name)
		if index:
			col_idx += 1
		# Set width of column to (column_length + margin)
		sheet.column_dimensions[openpyxl.utils.cell.get_column_letter(col_idx+1)].width = column_length * length_factor + margin
	# Compute column width of index column (if enabled)
	if index: # If the index column is being exported
		index_length =  max(df.index.map(_to_str).map(text_length).max(), text_length(df.index.name))
		sheet.column_dimensions["A"].width = index_length * length_factor + margin


if __name__ == "__main__":
	start = time.time()
	print('> No. of BOOTSTRAP ITERATIONS: {0}'.format(ITER_BOOTS))
	print('> No. of SEARCH ITERATIONS for CYTON FITTING: {0}'.format(ITER_SEARCH))

	########### DATA ###########
	DATA_FILES = [
		## Figure 5 - Complex interaction
		'DV24_001 CsA MPA OTI BimKO.xlsx',
		'DV23_004 DEX+CsA.xlsx',
	]
	KEYS = [os.path.splitext(os.path.basename(data_key))[0] for data_key in DATA_FILES]
	df = parse_data('./data', DATA_FILES)

	pos, inputs = 0, []
	for key in KEYS:
		reader = df[key]['reader']
		for icnd, cond in enumerate(reader.condition_names):
			if cond == 'US' or cond == 'unstimulated':
				pass
			inputs.append((key, df[key], reader, icnd, pos))
			pos += 1

	tqdm.tqdm.set_lock(mp.RLock())  # for managing output contention
	p = mp.Pool(initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),))
	with tqdm.tqdm(total=len(inputs), desc="Data Files", position=0) as pbar:
		for i, _ in enumerate(p.imap_unordered(fit, inputs)):
			pbar.update()
	p.close()
	p.join()

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)
	now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f"> DONE FITTING ! {now}")
	print("> Elapsed Time = {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))