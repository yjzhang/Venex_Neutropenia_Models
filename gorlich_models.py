
import numpy as np

import pymc as pm

from pytensor.compile.ops import as_op
import pytensor.tensor as pt



from new_patient_model import generate_forward_function

from tellurium_model_fitting import generate_objective_function_multiple_times


#######################
# model based on Banck 2019
# Model source: https://link.springer.com/article/10.1186/s12918-019-0684-0

model_desc_banck_2019 = """
species $c1, $l1, $c2, $l2

in_ven_treatment = 0;
in_aza_treatment = 0;

// c1, c2 - healthy cells
c1' = 2*a_c*p_c*s*c1 - p_c*c1 - d*c1 - in_ven_treatment*k_ven*p_c*c1 - in_aza_treatment*k_aza*p_c*c1;

c2' = 2*p_c*c1 - 2*a_c*p_c*s*c1 - d_2c*c2 ;

// l1, l2 - leukemic cells
l1' = 2*a_l*p_l*s*l1 - p_l*l1 - d*l1 - in_ven_treatment*k_ven*p_l*l1 - in_aza_treatment*k_aza*p_l*l1;

l2' = 2*p_l*l1 - 2*a_l*p_l*s*l1 - d_2l*l2 - d*l2;

x := c1 + l1 + l2;

s := 1/(1 + k_c*c2);

d := 10^-10*piecewise(0, x <= 4*10^9, x - 4*10^9);

// parameter values

// healthy hsc
a_c = 0.87;
p_c = 0.42;
d_2c = 2.3;
// slow progression
a_l = 0.9;
p_l = 0.2;
d_2l = 0.1;
// intermediate progression
// a_l = 0.92;
// p_l = 1;
// fast progression
// a_l = 1;
// p_l = 2;

// drug effect parameters - vary between 0 and 10
k_ven = 5;
k_aza = 5;

// initial conditions - in cells/kg
bw = 80;
c1 = 2*10^9; // 2*10^9 healthy stem cells per kg
c2 = 3.9*10^9;

l1 = 1;
l2 = 0;

k_c = (2*a_c - 1)/(3.9*10^9);

Xblasts_obs := 100*(l1 + l2)/(c1 + l1 + l2);

l1_0 = 10;
Xwbc = 0;
"""

param_names_banck_2019 = ['a_l', 'p_l', 'k_ven', 'k_aza', 'l1_0']
param_bounds_banck_2019 = [(0, 1), (0, 2), (0, 10), (0, 10), (0, 15)]

# for model fitting, maybe we could do a 2-step procedure? first, setting the drug effects to 0, find the 


def initialization_fn_banck_2019(model, param_vals):
    # initializing some of the model params to equilibrium values?
    l1_0 = param_vals[-1]
    model.setValue('l1_0', l1_0)
    model.setValue('l1', l1_0*10**9)
    return model


def generate_residual_function_blasts_only(model, blasts, n_samples=None, params_to_fit=None, initialization=None):
    """
    Generates a residual function using leukocyte and blast data (assuming that the cycle info is already incorporated into the model)
    """
    if initialization is None:
        initialization = initialization_fn_banck_2019
    if params_to_fit is None:
        params_to_fit = param_names_banck_2019
    # n_samples should be max time * 10?
    if n_samples is None:
        n_samples = blasts.iloc[:,0].max()*20
        print('n_samples:', n_samples)
    resid_function = generate_objective_function_multiple_times(model,
                                    {'Xblasts_obs': blasts.iloc[:,1].to_numpy()},
                                    {'Xblasts_obs': blasts.iloc[:,0].to_numpy()},
                                    params_to_fit=params_to_fit,
                                    use_val_ranges=False,
                                    set_initial_vals=False,
                                    n_samples=n_samples,
                                    var_weights=None,
                                    #global_param_vals=model_params,
                                    time_range=None,
                                    initialization_fn=initialization,
                                    return_values=True,
                                    handle_errors=True,
                                    print_errors=True)
    return resid_function


def generate_forward_function_blasts_only(model, blasts, n_samples=None, params_to_fit=None, initialization=None):
    ode_out = generate_residual_function_blasts_only(model, blasts, n_samples=n_samples, params_to_fit=params_to_fit,
          initialization=initialization)
    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def pytensor_forward_model(theta):
        output = ode_out(theta)
        blasts = np.array(output['Xblasts_obs'])
        return blasts
    return pytensor_forward_model



def build_pm_model_banck_2019(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_banck_2019
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    #print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    #print('wbc_data:', wbc_data)
    #print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            a_l = pm.TruncatedNormal("a_l", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            p_l = pm.TruncatedNormal("p_l", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            k_ven = pm.TruncatedNormal("k_ven", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            k_aza = pm.TruncatedNormal("k_aza", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
        else:
            a_l = pm.Uniform("a_l", lower=0, upper=1, initval=theta[0])
            p_l = pm.Uniform("p_l", lower=0, upper=2, initval=theta[1])
            k_ven = pm.Uniform("k_ven", lower=0, upper=10, initval=theta[2])
            k_aza = pm.Uniform("k_aza", lower=0, upper=10, initval=theta[3])
        l1_0 = pm.Uniform("l1_0", lower=0, upper=15, initval=theta[4])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([a_l, p_l,
                                                                k_ven, k_aza,
                                                                l1_0]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm


def build_pm_model_banck_2019_blast_only(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_banck_2019
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    #print('theta:', theta)
    pytensor_forward_function = generate_forward_function_blasts_only(model, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    #print('wbc_data:', wbc_data)
    #print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            a_l = pm.TruncatedNormal("a_l", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            p_l = pm.TruncatedNormal("p_l", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            k_ven = pm.TruncatedNormal("k_ven", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            k_aza = pm.TruncatedNormal("k_aza", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
        else:
            a_l = pm.Uniform("a_l", lower=0, upper=1, initval=theta[0])
            p_l = pm.Uniform("p_l", lower=0, upper=2, initval=theta[1])
            k_ven = pm.Uniform("k_ven", lower=0, upper=10, initval=theta[2])
            k_aza = pm.Uniform("k_aza", lower=0, upper=10, initval=theta[3])
        l1_0 = pm.Uniform("l1_0", lower=0, upper=15, initval=theta[4])
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([a_l, p_l,
                                                                k_ven, k_aza,
                                                                l1_0]))
        # split up the blasts and WBCs
        blast_ode = ode_solution[:len(blast_data)]

        # likelihood
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm

