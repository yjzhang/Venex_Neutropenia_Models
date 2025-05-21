# TODO: implement a thread pool with multiprocessing for systematic comparisons...

import os
import multiprocessing
import traceback

import numpy as np
import pandas as pd
import time

from new_patient_model import new_model_desc_no_leno, build_pm_model_from_dataframes, fit_model_mc, extract_data_from_tables_new,\
        run_model, plot_data, plot_runs, calculate_errors, split_train_test, generate_forward_function,\
        plot_runs_area, param_names_with_initial, initialization_fn_with_initial_2,\
        param_names_with_initial_bounds, split_cycles

from new_patient_additional_model_descs import model_desc_m4a_cytokine_independent,\
        param_names_m4a, param_names_bounds_m4a, initialization_fn_m4a_2, build_pm_model_m4a,\
        model_desc_m4b_direct_inhibition, param_names_m4b, param_names_bounds_m4b, initialization_fn_m4b_2,\
        build_pm_model_m4b, model_desc_m4c_direct_inhibition_independent_blasts,\
        param_names_m4b_wbc_only, initialization_fn_m4b_2_wbc_only, param_names_bounds_m4b_wbc_only,\
        build_pm_model_m4b_wbc_only, model_desc_m4b_wbc_only

from simplified_models import model_desc_m2, build_pm_model_m2,\
        initialization_fn_m2_2, param_names_m2_with_b0, generate_dosing_component_m2,\
        param_bounds_m2,\
        model_desc_m2_wbc_only, build_pm_model_m2_wbc_only, initialization_fn_m2_wbc_only, \
        param_names_m2_wbc_only, param_bounds_m2_wbc_only,\
        model_desc_m2b, build_pm_model_m2b, initialization_fn_m2b_2,\
        param_names_m2b, model_desc_m2c, build_pm_model_m2c, initialization_fn_m2c_2,\
        param_names_m2c, param_bounds_m2b, param_bounds_m2c,\
        model_desc_m2b_wbc_only, build_pm_model_m2b_wbc_only,\
        initialization_fn_m2b_2_wbc_only,\
        param_names_m2b_wbc_only, param_bounds_m2b_wbc_only,\
        model_desc_m2d, param_names_m2d, param_bounds_m2d, initialization_fn_m2d_2,\
        build_pm_model_m2d,\
        model_desc_m2e,\
        model_desc_m2f, param_names_m2f, param_bounds_m2f, initialization_fn_m2f_2,\
        build_pm_model_m2f,\
        model_desc_m2cs, param_names_m2cs, param_bounds_m2cs, initialization_fn_m2cs_2,\
        build_pm_model_m2cs,\
        model_desc_m2_blasts_only, param_names_m2_blasts_only,\
        param_bounds_m2_blasts_only, initialization_fn_m2_blasts_only,\
        build_pm_model_m2_blasts_only,\
        model_desc_m2b_blast_only, param_names_m2b_blast_only,\
        param_bounds_m2b_blast_only, initialization_fn_m2b_blast_only,\
        build_pm_model_m2b_blast_only

from hoffmann_like_models import model_desc_hoffmann_like_blast_only,\
        model_desc_hoffmann_like, param_names_hoffmann_like_blast_only,\
        param_bounds_hoffmann_like_blast_only, initialization_fn_hoffmann_like_blast_only,\
        build_pm_model_hoffmann_like_blast_only, param_names_hoffmann_like,\
        param_bounds_hoffmann_like, initialization_fn_hoffmann_like,\
        generate_dosing_component_hoffmann_like, build_pm_model_hoffmann_like,\
        model_desc_hoffmann_like_blast_only_ven_only,\
        param_names_hoffmann_like_blast_only_ven_only,\
        param_bounds_hoffmann_like_blast_only_ven_only,\
        initialization_fn_hoffmann_like_blast_only_ven_only,\
        build_pm_model_hoffmann_like_blast_only_ven_only


from gorlich_models import model_desc_banck_2019,\
        param_names_banck_2019, param_bounds_banck_2019,\
        initialization_fn_banck_2019, build_pm_model_banck_2019_blast_only

from find_map import fit_model_map, pybobyqa_wrapper

# this contains the key descriptions and functions for each model.
MODELS = {
        'm1b': {
            'param_names': param_names_with_initial,
            'model_desc': new_model_desc_no_leno,
            'initialization_fn': initialization_fn_with_initial_2,
            'build_pm_model': None,
            'param_bounds': param_names_with_initial_bounds + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm2a': {
            'param_names': param_names_m2_with_b0,
            'model_desc': model_desc_m2,
            'initialization_fn': initialization_fn_m2_2,
            'build_pm_model': build_pm_model_m2,
            'param_bounds': param_bounds_m2 + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2a_wbc_only': {
            'param_names': param_names_m2_wbc_only,
            'model_desc': model_desc_m2_wbc_only,
            'initialization_fn': initialization_fn_m2_wbc_only,
            'build_pm_model': build_pm_model_m2_wbc_only,
            'param_bounds': param_bounds_m2_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2a_w': {
            'param_names': param_names_m2_wbc_only,
            'model_desc': model_desc_m2_wbc_only,
            'initialization_fn': initialization_fn_m2_wbc_only,
            'build_pm_model': build_pm_model_m2_wbc_only,
            'param_bounds': param_bounds_m2_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2b': {
            'param_names': param_names_m2b,
            'model_desc': model_desc_m2b,
            'initialization_fn': initialization_fn_m2b_2,
            'build_pm_model': build_pm_model_m2b,
            'param_bounds': param_bounds_m2b + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2b_wbc_only': {
            'param_names': param_names_m2b_wbc_only,
            'model_desc': model_desc_m2b_wbc_only,
            'initialization_fn': initialization_fn_m2b_2_wbc_only,
            'build_pm_model': build_pm_model_m2b_wbc_only,
            'param_bounds': param_bounds_m2b_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2b_w': {
            'param_names': param_names_m2b_wbc_only,
            'model_desc': model_desc_m2b_wbc_only,
            'initialization_fn': initialization_fn_m2b_2_wbc_only,
            'build_pm_model': build_pm_model_m2b_wbc_only,
            'param_bounds': param_bounds_m2b_wbc_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': True
        },
        'm2c': {
            'param_names': param_names_m2c,
            'model_desc': model_desc_m2c,
            'initialization_fn': initialization_fn_m2c_2,
            'build_pm_model': build_pm_model_m2c,
            'param_bounds': param_bounds_m2c + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2cs': {
            'param_names': param_names_m2cs,
            'model_desc': model_desc_m2cs,
            'initialization_fn': initialization_fn_m2cs_2,
            'build_pm_model': build_pm_model_m2cs,
            'param_bounds': param_bounds_m2cs + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2d': {
            'param_names': param_names_m2d,
            'model_desc': model_desc_m2d,
            'initialization_fn': initialization_fn_m2d_2,
            'build_pm_model': build_pm_model_m2d,
            'param_bounds': param_bounds_m2d + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2e': {
            'param_names': param_names_m2b,
            'model_desc': model_desc_m2e,
            'initialization_fn': initialization_fn_m2b_2,
            'build_pm_model': build_pm_model_m2b,
            'param_bounds': param_bounds_m2b + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2f': {
            'param_names': param_names_m2f,
            'model_desc': model_desc_m2f,
            'initialization_fn': initialization_fn_m2f_2,
            'build_pm_model': build_pm_model_m2f,
            'param_bounds': param_bounds_m2f + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False
        },
        'm2_blasts_only': {
            'param_names': param_names_m2_blasts_only,
            'model_desc': model_desc_m2_blasts_only,
            'initialization_fn': initialization_fn_m2_blasts_only,
            'build_pm_model': build_pm_model_m2_blasts_only,
            'param_bounds': param_bounds_m2_blasts_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False,
            'blasts_only': True
        },
        'm2b_blast_only': {
            'param_names': param_names_m2b_blast_only,
            'model_desc': model_desc_m2b_blast_only,
            'initialization_fn': initialization_fn_m2b_blast_only,
            'build_pm_model': build_pm_model_m2b_blast_only,
            'param_bounds': param_bounds_m2b_blast_only + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False,
            'blasts_only': True
        },
        'm4a': {
            'param_names': param_names_m4a,
            'model_desc': model_desc_m4a_cytokine_independent,
            'initialization_fn': initialization_fn_m4a_2,
            'build_pm_model': build_pm_model_m4a,
            'param_bounds': param_names_bounds_m4a + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm4b': {
            'param_names': param_names_m4b,
            'model_desc': model_desc_m4b_direct_inhibition,
            'initialization_fn': initialization_fn_m4b_2,
            'build_pm_model': build_pm_model_m4b,
            'param_bounds': param_names_bounds_m4b + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'm4b_wbc_only': {
            'param_names': param_names_m4b_wbc_only,
            'model_desc': model_desc_m4b_wbc_only,
            'initialization_fn': initialization_fn_m4b_2_wbc_only,
            'build_pm_model': build_pm_model_m4b_wbc_only,
            'param_bounds': param_names_bounds_m4b_wbc_only + [(0, 10)],
            'dosing_component': None,
            'wbc_only': True
        },
        'm4b_w': {
            'param_names': param_names_m4b_wbc_only,
            'model_desc': model_desc_m4b_wbc_only,
            'initialization_fn': initialization_fn_m4b_2_wbc_only,
            'build_pm_model': build_pm_model_m4b_wbc_only,
            'param_bounds': param_names_bounds_m4b_wbc_only + [(0, 10)],
            'dosing_component': None,
            'wbc_only': True
        },
        'm4c': {
            'param_names': param_names_m4b,
            'model_desc': model_desc_m4c_direct_inhibition_independent_blasts,
            'initialization_fn': initialization_fn_m4b_2,
            'build_pm_model': build_pm_model_m4b,
            'param_bounds': param_names_bounds_m4b + [(0, 10), (0, 10)],
            'dosing_component': None,
            'wbc_only': False
        },
        'hoffmann_like_blast_only': {
            'param_names': param_names_hoffmann_like_blast_only,
            'model_desc': model_desc_hoffmann_like_blast_only,
            'initialization_fn': initialization_fn_hoffmann_like_blast_only,
            'build_pm_model': build_pm_model_hoffmann_like_blast_only,
            'param_bounds': param_bounds_hoffmann_like_blast_only + [(0, 10)],
            'dosing_component': generate_dosing_component_hoffmann_like,
            'wbc_only': False,
            'blasts_only': True
        },
        'hoffmann_like_blast_only_ven_only': {
            'param_names': param_names_hoffmann_like_blast_only_ven_only,
            'model_desc': model_desc_hoffmann_like_blast_only_ven_only,
            'initialization_fn': initialization_fn_hoffmann_like_blast_only_ven_only,
            'build_pm_model': build_pm_model_hoffmann_like_blast_only_ven_only,
            'param_bounds': param_bounds_hoffmann_like_blast_only_ven_only + [(0, 10)],
            'dosing_component': generate_dosing_component_hoffmann_like,
            'wbc_only': False,
            'blasts_only': True
        },
        'hoffmann_like': {
            'param_names': param_names_hoffmann_like,
            'model_desc': model_desc_hoffmann_like,
            'initialization_fn': initialization_fn_hoffmann_like,
            'build_pm_model': build_pm_model_hoffmann_like,
            'param_bounds': param_bounds_hoffmann_like + [(0, 10), (0, 10)],
            'dosing_component': generate_dosing_component_hoffmann_like,
            'wbc_only': False,
            'blasts_only': False
        },
        'banck_2019_blast_only': {
            'param_names': param_names_banck_2019,
            'model_desc': model_desc_banck_2019,
            'initialization_fn': initialization_fn_banck_2019,
            'build_pm_model': build_pm_model_banck_2019_blast_only,
            'param_bounds': param_bounds_banck_2019 + [(0, 10)],
            'dosing_component': generate_dosing_component_m2,
            'wbc_only': False,
            'blasts_only': True
        },
}

def build_model(model_name, cycle_days, blood_counts, bm_blasts, patient_id,
        params=None):
    """
    Returns a tellurium model and a pymc model.
    """
    if blood_counts is None:
        blood_counts = pd.DataFrame(columns=['Pseudonym', 'days_lab', 'b_neut'])
    if bm_blasts is None:
        bm_blasts = pd.DataFrame(columns=['Pseudonym', 'days_from_bm', 'bm_blasts'])
    model_functions = MODELS[model_name]
    model_desc = model_functions['model_desc']
    build_pm_model = model_functions['build_pm_model']
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    dosing_component = None
    if 'dosing_component' in model_functions:
        dosing_component = model_functions['dosing_component']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    blasts_only = False
    if 'blasts_only' in model_functions:
        blasts_only = model_functions['blasts_only']
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
            bm_blasts, cycle_days, patient_id, use_neut=True)
    # build model
    param_vals = params
    if isinstance(params, dict):
        params = [params[k] for k in param_names]
    # ugh...
    if param_vals is not None and not isinstance(param_vals, dict):
        param_vals = {k: v for k, v in zip(param_names, param_vals)}
    te_model, pm_model = build_pm_model_from_dataframes(cycle_info, leuk_table, blast_table,
            param_vals=param_vals,
            use_neut=True, use_b0='B0' in param_names,
            model_desc=model_desc,
            build_model_function=build_pm_model,
            initialization=initialization_fn,
            params_to_fit=param_names,
            dosing_component_function=dosing_component,
            uniform_prior=True,
            use_initial_guess=True,
            use_blasts=use_blasts,
            use_blasts_interpolation=not use_blasts,
            theta=params,
            blasts_only=blasts_only
            )
    return te_model, pm_model

# run a single model with a specific set of parameters/on a specific patient
def run_model_on_patient(model_name, params, cycle_days, blood_counts, bm_blasts, patient_id,
        max_time=None, n_samples_per_day=20, selections=None,
        return_errors=False, error_fn='rmse'):
    """
    Runs a model with a given set of parameters.

    Args:
        - model_name: a string - has to be a key in MODELS
        - params - list of parameters, or a dict of parameters as output from fit_model_results
        - blood_counts: table of blood cell measurements. Can be None. Required columns: 'Pseudonym', 'days_lab', 'b_neut'
        - bm_blasts: table of blast measurements. Can be None. Required columns: 'Pseudonym', 'days_bm', 'bm_blasts'
        - cycle_days: table of drug treatment cycles. Required columns: 'days_aza_start', 'days_ven_start', 'aza_days', 'venetoclax_days', 'aza_dose_mg', 'venetoclax_dose_mg'
        - patient_id: ID for the given patient

    Optional args:
        - max_time - maximum time to run the simulation to
        - n_samples_per_day
        - selections: list of model variables to return.
        - return_errors: whether or not to return the model errors.
        - error_fn: error function - default: 'rmse'

    Returns:
        - results - a keyed array containing fields for 'time', 'Xwbc', 'Xblasts_obs'
    """
    # if blood_counts and bm_blasts are not present or incomplete:
    # create empty tables
    if blood_counts is None:
        blood_counts = pd.DataFrame(columns=['Pseudonym', 'days_lab', 'b_neut'])
    if bm_blasts is None:
        bm_blasts = pd.DataFrame(columns=['Pseudonym', 'days_from_bm', 'bm_blasts'])
    model_functions = MODELS[model_name]
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    blasts_only = False
    if 'blasts_only' in model_functions:
        blasts_only = model_functions['blasts_only']
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
            bm_blasts, cycle_days, patient_id, use_neut=True)
    # build model
    if isinstance(params, dict):
        params = [params[k] for k in param_names]
    te_model, pm_model = build_model(model_name, cycle_days, blood_counts,
            bm_blasts, patient_id, params=None)
    # run model
    rmse_leuk, rmse_blasts, results = calculate_errors(te_model, params,
            cycle_info, leuk_table, blast_table,
            use_neut=True,
            error_fn=error_fn,
            initialization=initialization_fn,
            params_to_fit=param_names,
            wbc_only=not use_blasts,
            blasts_only=blasts_only,
            max_time=max_time,
            ode_samples_per_day=n_samples_per_day)
    if return_errors:
        return rmse_leuk, rmse_blasts, results
    return results

# run a single model on a patient, with the given parameters as a starting point for MCMC. Returns a sample of 200 points.
def run_model_on_patient_mcmc(model_name, params, cycle_days, blood_counts, bm_blasts, patient_id,
        max_time=None, n_samples_per_day=20, selections=None,
        return_errors=False, error_fn='rmse', n_samples=5000,
        n_samples_returned=200, progress_filename=None):
    """
    Runs a model with a given set of parameters.

    Args:
        - model_name: a string - has to be a key in MODELS
        - params - list of parameters, or a dict of parameters as output from fit_model_results
        - blood_counts: table of blood cell measurements. Can be None. Required columns: 'Pseudonym', 'days_lab', 'b_neut'
        - bm_blasts: table of blast measurements. Can be None. Required columns: 'Pseudonym', 'days_bm', 'bm_blasts'
        - cycle_days: table of drug treatment cycles. Required columns: 'days_aza_start', 'days_ven_start', 'aza_days', 'venetoclax_days', 'aza_dose_mg', 'venetoclax_dose_mg'
        - patient_id: ID for the given patient

    Optional args:
        - max_time - maximum time to run the simulation to
        - n_samples_per_day
        - selections: list of model variables to return.
        - return_errors: whether or not to return the model errors.
        - error_fn: error function - default: 'rmse'
        - n_samples - this is the number of mcmc draws per chain.
        - n_samples_returned - this is the number of samples to return.

    Returns:
        - pymc_posterior_means, trace
    """
    import arviz as az
    # if blood_counts and bm_blasts are not present or incomplete:
    # create empty tables
    if blood_counts is None:
        blood_counts = pd.DataFrame(columns=['Pseudonym', 'days_lab', 'b_neut'])
    if bm_blasts is None:
        bm_blasts = pd.DataFrame(columns=['Pseudonym', 'days_from_bm', 'bm_blasts'])
    model_functions = MODELS[model_name]
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    blasts_only = False
    if 'blasts_only' in model_functions:
        blasts_only = model_functions['blasts_only']
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
                                                                             bm_blasts, cycle_days, patient_id, use_neut=True)
    # build model
    if isinstance(params, dict):
        params = [params[k] for k in param_names]
    te_model, pm_model = build_model(model_name, cycle_days, blood_counts,
            bm_blasts, patient_id, params)
    # run MCMC
    # use the given initial point
    pymc_posterior_means, trace = fit_model_mc(pm_model, draws=int(n_samples), params_to_fit=param_names)
    trace_df = az.extract(trace, num_samples=int(n_samples_returned)).to_dataframe()
    # run every trace - see plot_runs in new_patient_model.py
    all_outputs = []
    for _, r in trace_df.iterrows():
        vals = r[param_names].to_numpy()
        try:
            rmse_leuk, rmse_blasts, results = calculate_errors(te_model, vals, cycle_info, leuk_table, blast_table,
                                                       use_neut=True,
                                                       error_fn=error_fn,
                                                       initialization=initialization_fn,
                                                       params_to_fit=param_names,
                                                       wbc_only=not use_blasts,
                                                       blasts_only=blasts_only,
                                                       max_time=max_time,
                                                       ode_samples_per_day=n_samples_per_day)
            all_outputs.append(results)
        except:
            pass
    # all outputs is a list of all of the trajectories
    return pymc_posterior_means, trace_df, all_outputs


def fit_model_results(blood_counts, bm_blasts, cycle_days, patient_id,
    model_name, prior_params=None, uniform_prior=True, use_mcmc=False,
    progress_filename=None, map_kwargs=None):
    """
    Fits a model on a single patient, returning the model parameters
    and the fit error/RMSE.

    Args:
        - blood_counts: table of blood cell measurements
        - bm_blasts: table of blast measurements
        - cycle_days: table of drug treatment cycles
        - patient_id: ID for the given patient
        - model_name: a string, representing a key in MODELS
        - progress_filename: file to write progress
        - map_kwargs: dict of parameters to pass to the optimizer in fit_model_map

    Returns:
        map_rmse, map_params_dict
    """
    print('Patient ID:', patient_id)
    model_functions = MODELS[model_name]
    model_desc = model_functions['model_desc']
    build_pm_model = model_functions['build_pm_model']
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    param_bounds = model_functions['param_bounds']
    dosing_component = None
    if 'dosing_component' in model_functions:
        dosing_component = model_functions['dosing_component']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    blasts_only = False
    if 'blasts_only' in model_functions:
        blasts_only = model_functions['blasts_only']
    t0 = time.time()
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
                                                                             bm_blasts, cycle_days, patient_id, use_neut=True)
    te_model, pm_model = build_pm_model_from_dataframes(cycle_info, leuk_table, blast_table,
            use_neut=True, use_b0=True,
            model_desc=model_desc,
            build_model_function=build_pm_model,
            initialization=initialization_fn,
            params_to_fit=param_names,
            dosing_component_function=dosing_component,
            uniform_prior=uniform_prior,
            use_initial_guess=True,
            use_blasts=use_blasts,
            use_blasts_interpolation=not use_blasts,
            theta=prior_params,
            blasts_only=blasts_only
            )
    try:
        if use_mcmc:
            posterior_means, trace = fit_model_mc(pm_model, params_to_fit=param_names)
            # TODO: what to do with the trace???
            map_params = posterior_means
        else:
            if map_kwargs is None:
                map_kwargs = {'maxfun': 20000}
            map_params = fit_model_map(pm_model, params_to_fit=param_names, maxeval=50000,
                                       method=pybobyqa_wrapper,
                                       bounds=param_bounds,
                                       options=map_kwargs,
                                       progressbar=False,
                                       progress_filename=progress_filename)
        print('time:', time.time() - t0)
        map_params_dict = {k: v for k, v in zip(param_names, map_params)}
        rmse_leuk, rmse_blasts, results = calculate_errors(te_model, map_params, cycle_info, leuk_table, blast_table,
                                                              use_neut=True,
                                                              error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts,
                                                            blasts_only=blasts_only)
        rmse_results = {'blasts_train': rmse_blasts, 'leuk_train': rmse_leuk}
        map_rmse = rmse_results
        print('RMSE results:', rmse_results)
        print('nRMSE_blasts:', rmse_blasts/blast_table.bm_blasts.std(), 'nRMSE_neut:', rmse_leuk/leuk_table.b_neut.std())
    except Exception as e:
        print('Unable to run patient')
        print(e)
        print(traceback.format_exc())
        map_rmse = {'blasts_train': np.inf, 'leuk_train': np.inf}
        map_params_dict = {k: np.nan for k in param_names}
    return map_rmse, map_params_dict


def fit_model_train_test(blood_counts, bm_blasts, cycle_days, patient_id,
        model_name, n_cycles_train=4, n_cycles_test=2,
        prior_params=None, uniform_prior=True, use_mcmc=False):
    print('Patient ID:', patient_id)
    model_functions = MODELS[model_name]
    model_desc = model_functions['model_desc']
    build_pm_model = model_functions['build_pm_model']
    initialization_fn = model_functions['initialization_fn']
    param_names = model_functions['param_names']
    param_bounds = model_functions['param_bounds']
    dosing_component = None
    if 'dosing_component' in model_functions:
        dosing_component = model_functions['dosing_component']
    use_blasts = True
    if 'wbc_only' in model_functions:
        use_blasts = not model_functions['wbc_only']
    blasts_only = False
    if 'blasts_only' in model_functions:
        blasts_only = model_functions['blasts_only']
    t0 = time.time()
    cycle_info, leuk_table, blast_table = extract_data_from_tables_new(blood_counts,
            bm_blasts, cycle_days, patient_id, use_neut=True)
    if len(cycle_info) < n_cycles_train + n_cycles_test:
        print('Too few cycles in patient')
        map_rmse = {'blasts_train': np.inf, 'leuk_train': np.inf,
                'blasts_test': np.inf, 'leuk_test': np.inf,
                'blasts_additional': np.inf, 'leuk_additional': np.inf}
        map_params_dict = {k: np.nan for k in param_names}
        return map_rmse, map_params_dict
    # split cycles into train and test
    leuk_train, leuk_test, blast_train, blast_test, leuk_remainder, blast_remainder = split_cycles(leuk_table, blast_table, cycle_info, n_cycles_train, n_cycles_test)
    te_model, pm_model = build_pm_model_from_dataframes(cycle_info, leuk_train, blast_train,
            use_neut=True, use_b0=True,
            model_desc=model_desc,
            build_model_function=build_pm_model,
            initialization=initialization_fn,
            params_to_fit=param_names,
            dosing_component_function=dosing_component,
            uniform_prior=uniform_prior,
            use_initial_guess=True,
            use_blasts=use_blasts,
            theta=prior_params
            )
    try:
        if use_mcmc:
            map_params, trace = fit_model_mc(pm_model, params_to_fit=param_names)
        else:
            map_params = fit_model_map(pm_model, params_to_fit=param_names, maxeval=50000,
                                       method=pybobyqa_wrapper,
                                       bounds=param_bounds,
                                       options={'maxfun':20000},
                                       progressbar=False)
        print('time:', time.time() - t0)
        map_params_dict = {k: v for k, v in zip(param_names, map_params)}
        rmse_leuk_train, rmse_blasts_train, _ = calculate_errors(te_model, map_params, cycle_info, leuk_train, blast_train,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts,
                                                            blasts_only=blasts_only)
        rmse_leuk_test, rmse_blasts_test, _ = calculate_errors(te_model, map_params, cycle_info, leuk_test, blast_test,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts,
                                                            blasts_only=blasts_only)
        leuk_additional = pd.concat([leuk_test, leuk_remainder])
        blast_additional = pd.concat([blast_test, blast_remainder])
        rmse_leuk_additional, rmse_blasts_additional, _ = calculate_errors(te_model, map_params, cycle_info,
                                                            leuk_additional, blast_additional,
                                                            use_neut=True,
                                                            error_fn='rmse',
                                                            initialization=initialization_fn,
                                                            params_to_fit=param_names,
                                                            wbc_only=not use_blasts,
                                                            blasts_only=blasts_only)
        rmse_results = {'blasts_train': rmse_blasts_train, 'leuk_train': rmse_leuk_train,
                'blasts_test': rmse_blasts_test, 'leuk_test': rmse_leuk_test,
                'blasts_additional': rmse_blasts_additional, 'leuk_additional': rmse_leuk_additional}
        map_rmse = rmse_results
        print('RMSE results:', rmse_results)
        print('nRMSE_blasts:', rmse_blasts_train/blast_table.bm_blasts.std(), 'nRMSE_neut:', rmse_leuk_train/leuk_table.b_neut.std())
    except Exception as e:
        print('Unable to run patient')
        print(e)
        print(traceback.format_exc())
        map_rmse = {'blasts_train': np.inf, 'leuk_train': np.inf,
                'blasts_test': np.inf, 'leuk_test': np.inf,
                'blasts_additional': np.inf, 'leuk_additional': np.inf}
        map_params_dict = {k: np.nan for k in param_names}
    return map_rmse, map_params_dict



def run_all_patients(model_name, params_filename=None, rmse_filename=None,
        n_threads=None, train_test=False,
        n_cycles_train=4, n_cycles_test=2,
        uniform_prior=True,
        use_mcmc=False,
        map_kwargs=None):
    """
    For a given model, runs all patients using a multiprocessing pool...
    """
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test, None, uniform_prior, use_mcmc) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, None, uniform_prior, use_mcmc, None, map_kwargs) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)


def run_all_patients_1_vs_rest(model_name, prior_params_filename,
        params_filename=None, rmse_filename=None, n_threads=None,
        uniform_prior=False, train_test=False,
        n_cycles_train=4, n_cycles_test=2,
        map_kwargs=None):
    """
    Runs all patients using a prior parameter set constructed from all other patients.
    """
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    # load prior params
    prior_params = pd.read_csv(prior_params_filename, index_col=0)
    prior_params_dict = {}
    # for every patient, create a prior based on all other patients.
    for p in patient_ids:
        other_patient_params = prior_params[prior_params.index != p].mean(0)
        prior_params_dict[p] = other_patient_params.to_list()
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)

# TODO: priors constructed using linear regression instead of stuff
def run_all_patients_clinical_prior(model_name, prior_params_filename,
        clinical_data_filename='patient_data_venex/ven_responses_052023.txt',
        clinical_features=[],
        params_filename=None, rmse_filename=None, n_threads=None,
        uniform_prior=False, train_test=False,
        n_cycles_train=4, n_cycles_test=2):
    """
    Runs all patients using a prior parameter set constructed from clinical data.

    For each patient, we will construct a linear regression model using all
    other patients to predict each model parameter.
    """
    import statsmodels.api as sm
    # load data
    blood_counts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Blood_counts')
    bm_blasts = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Bone_marrow_blasts')
    cycle_days = pd.read_excel('patient_data_venex/Ven_blood_counts_16042023.xlsx', sheet_name='Cycle_days')
    patient_ids = np.loadtxt('id_samples_3_cycles.txt', dtype=int)
    # load prior params
    prior_params = pd.read_csv(prior_params_filename, index_col=0)
    # load clinical data
    clinical_data = pd.read_csv(clinical_data_filename)
    # TODO: build a model using statsmodels

    prior_params_dict = {}
    # for every patient, create a prior based on all other patients.
    for p in patient_ids:
        other_patient_params = prior_params[prior_params.index != p].mean(0)
        prior_params_dict[p] = other_patient_params.to_list()
    if n_threads is None:
        n_threads = os.cpu_count()
    pool = multiprocessing.Pool(n_threads)
    if train_test:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, n_cycles_train, n_cycles_test, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_train_test, args, int(len(patient_ids)/n_threads))
    else:
        args = [(blood_counts, bm_blasts, cycle_days, p, model_name, prior_params_dict[p], uniform_prior) for p in patient_ids]
        results = pool.starmap(fit_model_results, args, int(len(patient_ids)/n_threads))
    pool.close()
    pool.terminate()
    rmse_results = {patient_id: x[0] for patient_id, x in zip(patient_ids, results)}
    param_results = {patient_id: x[1] for patient_id, x in zip(patient_ids, results)}
    rmse_data = pd.DataFrame(rmse_results).T
    param_data = pd.DataFrame(param_results).T
    if rmse_filename is None:
        if train_test:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_rmse_data.csv'
        else:
            rmse_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_rmse_data.csv'
    if params_filename is None:
        if train_test:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_train_test_mp_priors_param_data.csv'
        else:
            params_filename = f'systematic_comparison_results_simplified_models/{model_name}_mp_priors_param_data.csv'
    rmse_data.to_csv(rmse_filename)
    param_data.to_csv(params_filename)




if __name__ == '__main__':
    # comparing different optimizers
    #params_filename = 'systematic_comparison_results_different_optimizers/m2c_pybobyqa_fake_bounds_mp_param_data.csv'
    #rmse_filename = 'systematic_comparison_results_different_optimizers/m2c_pybobyqa_fake_bounds_mp_rmse_data.csv'
    #run_all_patients('m2c', n_threads=8, params_filename=params_filename,
    #        rmse_filename=rmse_filename,
    #        map_kwargs={'maxfun': 20000, 'fake_bounds': True})
    params_filename = 'systematic_comparison_results_different_optimizers/m2f_pybobyqa_fake_bounds_mp_param_data.csv'
    rmse_filename = 'systematic_comparison_results_different_optimizers/m2f_pybobyqa_fake_bounds_mp_rmse_data.csv'
    run_all_patients('m2f', n_threads=8, params_filename=params_filename,
            rmse_filename=rmse_filename,
            map_kwargs={'maxfun': 20000, 'fake_bounds': True})
    """
    run_all_patients('m2b', params_filename='systematic_comparison_results_simplified_models/m2b_mp_param_data.csv',
            rmse_filename='systematic_comparison_results_simplified_models/m2b_mp_rmse_data.csv',
            n_threads=8)
    run_all_patients('m2c', params_filename='systematic_comparison_results_simplified_models/m2c_mp_param_data.csv',
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_rmse_data.csv',
            n_threads=8)
    run_all_patients('m2d', n_threads=8)
    run_all_patients('m2b_wbc_only', n_threads=8)
    """
    #run_all_patients('m1b', n_threads=8)
    #run_all_patients('m4a', n_threads=8)
    #run_all_patients('m4b', n_threads=8)
    #run_all_patients('m2e', n_threads=8)
    #run_all_patients('m2f', n_threads=8)
    #run_all_patients('m4b_wbc_only', n_threads=8)
    # train-test
    """
    run_all_patients('m2f', train_test=True, n_cycles_train=1, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_rmse_data_1_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_param_data_1_cycle_train.csv')
    run_all_patients('m2f', train_test=True, n_cycles_train=2, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_rmse_data_2_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_param_data_2_cycle_train.csv')
    run_all_patients('m2f', train_test=True, n_cycles_train=3, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_rmse_data_3_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_param_data_3_cycle_train.csv')
    run_all_patients('m2f', train_test=True, n_cycles_train=4, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_rmse_data_4_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_param_data_4_cycle_train.csv')
    run_all_patients('m2f', train_test=True, n_cycles_train=5, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_rmse_data_5_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2f_mp_train_test_param_data_5_cycle_train.csv')
    """
    """
    run_all_patients('m2c', train_test=True, n_cycles_train=1, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_1_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_1_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=2, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_2_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_2_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=3, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_3_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_3_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=4, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_4_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_4_cycle_train.csv')
    run_all_patients('m2c', train_test=True, n_cycles_train=5, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_5_cycle_train.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_5_cycle_train.csv')
    """
    # train-test with priors
    """
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=1, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_1_cycle_train_prior.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_1_cycle_train_prior.csv')
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=2, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_2_cycle_train_prior.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_2_cycle_train_prior.csv')
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=3, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_3_cycle_train_prior.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_3_cycle_train_prior.csv')
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=4, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_4_cycle_train_prior.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_4_cycle_train_prior.csv')
    run_all_patients_1_vs_rest('m2c', 'systematic_comparison_results_simplified_models/m2c_mp_param_data.csv', train_test=True, n_cycles_train=5, n_cycles_test=2,
            rmse_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_rmse_data_5_cycle_train_prior.csv',
            params_filename='systematic_comparison_results_simplified_models/m2c_mp_train_test_param_data_5_cycle_train_prior.csv')
    """
    # TODO: test run_model_on_patient_mcmc
