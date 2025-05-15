# Hoffmann-like models import numpy as np

import numpy as np
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

from tellurium_model_fitting import generate_objective_function_multiple_times
from new_patient_model import generate_forward_function


################################ blasts-only model


model_desc_hoffmann_like_blast_only = """
species $Hq; species $Lq; species $Ha; species $La; species $LkBd;

//rates
Hq' = Tqh*(1-((Lq+Hq)/Kq))*Ha - Tah*(1-((La+Ha)/Ka))*Hq; // Change in Quiencient Healthy Cells over time
Lq' = Tql*(1-((Lq+Hq)/Kq))*La - Tal*(1-((La+Ha)/Ka))*Lq; // Change in Quiencient Leukemic Cells over time
Ha' = Tah*(1-((La+Ha)/Ka))*Hq - Tqh*(1-((Lq+Hq)/Kq))*Ha + (Ph*(1-((La+Ha)/Ka))-c-Dh)*Ha; // Change in Activated Healthy Cells over time
La' = Tal*(1-((La+Ha)/Ka))*Lq - Tql*(1-((Lq+Hq)/Kq))*La + (Pl*(1-((La+Ha)/Ka))-c-Dl)*La; // Change in Activated Leukemic Cells over time

// TODO: blasts don't self-reproduce. They should be a separate layer of differentiation from the leukemic stem cells?

//parameters
Tqh = 0.2;
Tql = 0.002;
Kq = 10000; // Confused on what this Parameter should be
Tah = 0.01;
Tal = 0.25; // Guestimate from Figure 2a
Ka = 10000; // Confused on what this Parameter should be
Pl = 0.099; // Guestimate from Figure 2a
Ph = 1/25;
Dh = 1/30;
Dl = 1/30;

// parameters for the WBC component
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 2; //4.67; // unit: G/L
B0 = 2; //4.67;
kwbc = 2.3765; 

// kill factor
kf = 0.99;
kf_aza = 0.1;

// chemotherapy time periods
in_ven_treatment = 0;
in_aza_treatment = 0;
c := in_ven_treatment*kf + in_aza_treatment*kf_aza;

//initial values
Lq = 10000*.01; // Randomly Decided
Hq = 10000*.71; // Randomly Decided
La = 10000*.20; // Randomly Decided
Ha = 10000*.08; // Randomly Decided
//LkBd := log10(100*(La*100 + Lq*100)/(La + Ha + Lq + Hq));
La0 = La;
Lq0 = Lq;
Ha0 = Ha;
Hq0 = Hq;

Xblasts_obs := 100*(La + Lq)/(La + Ha + Lq + Hq);
"""

param_names_hoffmann_like_blast_only = ['Tal', 'Pl', 'kf', 'kf_aza', 'La0']
param_bounds_hoffmann_like_blast_only = [(0,1), (0,1), (0,1.5), (0,1.5), (0,9000)]

def initialization_fn_hoffmann_like_blast_only(model, param_vals):
    La = param_vals[4]
    Lq = La*0.05
    model.setValue('La', La)
    model.setValue('Lq', Lq)
    model.setValue('Hq', (10000 - (La+Lq))*0.9)
    model.setValue('Ha', (10000 - (La+Lq))*0.1)
    return model


def generate_residual_function_blasts_only(model, blasts, n_samples=None, params_to_fit=None, initialization=None):
    """
    Generates a residual function using leukocyte and blast data (assuming that the cycle info is already incorporated into the model)
    """
    if initialization is None:
        initialization = initialization_fn_hoffmann_like_blast_only
    if params_to_fit is None:
        params_to_fit = param_names_hoffmann_like_blast_only
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


def build_pm_model_hoffmann_like_blast_only(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_hoffmann_like_blast_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function_blasts_only(model, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            Tal = pm.TruncatedNormal("Tal", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            Pl = pm.TruncatedNormal("Pl", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            kf = pm.TruncatedNormal("kf", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            kf_aza = pm.TruncatedNormal("kf_aza", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            La0 = pm.TruncatedNormal('La0', mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
        else:
            Tal = pm.Uniform("Tal", lower=0, upper=1, initval=theta[0])
            Pl = pm.Uniform("Pl", lower=0, upper=1, initval=theta[1])
            kf = pm.Uniform("kf", lower=0, upper=2, initval=theta[2])
            kf_aza = pm.Uniform("kf_aza", lower=0, upper=2, initval=theta[3])
            La0 = pm.Uniform('La0', lower=0, upper=10000, initval=theta[4])
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, kf_aza, La0]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, kf_aza, La0]))
        
        # split up the blasts and WBCs
        blast_ode = ode_solution[:len(blast_data)]

        # likelihood
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm



################### ven only, no aza

model_desc_hoffmann_like_blast_only_ven_only = """
species $Hq; species $Lq; species $Ha; species $La; species $LkBd;

//rates
Hq' = Tqh*(1-((Lq+Hq)/Kq))*Ha - Tah*(1-((La+Ha)/Ka))*Hq; // Change in Quiencient Healthy Cells over time
Lq' = Tql*(1-((Lq+Hq)/Kq))*La - Tal*(1-((La+Ha)/Ka))*Lq; // Change in Quiencient Leukemic Cells over time
Ha' = Tah*(1-((La+Ha)/Ka))*Hq - Tqh*(1-((Lq+Hq)/Kq))*Ha + (Ph*(1-((La+Ha)/Ka))-c-Dh)*Ha; // Change in Activated Healthy Cells over time
La' = Tal*(1-((La+Ha)/Ka))*Lq - Tql*(1-((Lq+Hq)/Kq))*La + (Pl*(1-((La+Ha)/Ka))-c-Dl)*La; // Change in Activated Leukemic Cells over time

// TODO: blasts don't self-reproduce. They should be a separate layer of differentiation from the leukemic stem cells?

//parameters
Tqh = 0.2;
Tql = 0.002;
Kq = 10000; // Confused on what this Parameter should be
Tah = 0.01;
Tal = 0.25; // Guestimate from Figure 2a
Ka = 10000; // Confused on what this Parameter should be
Pl = 0.099; // Guestimate from Figure 2a
Ph = 1/25;
Dh = 1/30;
Dl = 1/30;

// parameters for the WBC component
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 2; //4.67; // unit: G/L
B0 = 2; //4.67;
kwbc = 2.3765; 

// kill factor
kf = 0.99;

// chemotherapy time periods
in_ven_treatment = 0;
in_aza_treatment = 0;
c := in_ven_treatment*kf;

//initial values
Lq = 10000*.01; // Randomly Decided
Hq = 10000*.71; // Randomly Decided
La = 10000*.20; // Randomly Decided
Ha = 10000*.08; // Randomly Decided
//LkBd := log10(100*(La*100 + Lq*100)/(La + Ha + Lq + Hq));
La0 = La;
Lq0 = Lq;
Ha0 = Ha;
Hq0 = Hq;

Xblasts_obs := 100*(La + Lq)/(La + Ha + Lq + Hq);
"""

param_names_hoffmann_like_blast_only_ven_only = ['Tal', 'Pl', 'kf', 'La0']
param_bounds_hoffmann_like_blast_only_ven_only = [(0,1), (0,1), (0,1.5), (0,9000)]

def initialization_fn_hoffmann_like_blast_only_ven_only(model, param_vals):
    La = param_vals[3]
    Lq = La*0.05
    model.setValue('La', La)
    model.setValue('Lq', Lq)
    model.setValue('Hq', (10000 - (La+Lq))*0.9)
    model.setValue('Ha', (10000 - (La+Lq))*0.1)
    return model


def generate_residual_function_blasts_only_ven_only(model, blasts, n_samples=None, params_to_fit=None, initialization=None):
    """
    Generates a residual function using leukocyte and blast data (assuming that the cycle info is already incorporated into the model)
    """
    if initialization is None:
        initialization = initialization_fn_hoffmann_like_blast_only_ven_only
    if params_to_fit is None:
        params_to_fit = param_names_hoffmann_like_blast_only_ven_only
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


def generate_forward_function_blasts_only_ven_only(model, blasts, n_samples=None, params_to_fit=None, initialization=None):
    ode_out = generate_residual_function_blasts_only_ven_only(model, blasts, n_samples=n_samples, params_to_fit=params_to_fit,
          initialization=initialization)
    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def pytensor_forward_model(theta):
        output = ode_out(theta)
        blasts = np.array(output['Xblasts_obs'])
        return blasts
    return pytensor_forward_model


def build_pm_model_hoffmann_like_blast_only_ven_only(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_hoffmann_like_blast_only_ven_only
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function_blasts_only_ven_only(model, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            Tal = pm.TruncatedNormal("Tal", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            Pl = pm.TruncatedNormal("Pl", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            kf = pm.TruncatedNormal("kf", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            La0 = pm.TruncatedNormal('La0', mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[3])
        else:
            Tal = pm.Uniform("Tal", lower=0, upper=1, initval=theta[0])
            Pl = pm.Uniform("Pl", lower=0, upper=1, initval=theta[1])
            kf = pm.Uniform("kf", lower=0, upper=2, initval=theta[2])
            La0 = pm.Uniform('La0', lower=0, upper=10000, initval=theta[3])
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, La0]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, La0]))
        
        # split up the blasts and WBCs
        blast_ode = ode_solution[:len(blast_data)]

        # likelihood
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm




################################# all model (not just blasts)

model_desc_hoffmann_like = """
species $Hq; species $Lq; species $Ha; species $La; species $LkBd;

//rates
Hq' = Tqh*(1-((Lq+Hq)/Kq))*Ha - Tah*(1-((La+Ha)/Ka))*Hq; // Change in Quiencient Healthy Cells over time
Lq' = Tql*(1-((Lq+Hq)/Kq))*La - Tal*(1-((La+Ha)/Ka))*Lq; // Change in Quiencient Leukemic Cells over time
Ha' = Tah*(1-((La+Ha)/Ka))*Hq - Tqh*(1-((Lq+Hq)/Kq))*Ha + (Ph*(1-((La+Ha)/Ka))-c-Dh)*Ha; // Change in Activated Healthy Cells over time
La' = Tal*(1-((La+Ha)/Ka))*Lq - Tql*(1-((Lq+Hq)/Kq))*La + (Pl*(1-((La+Ha)/Ka))-c-Dl)*La; // Change in Activated Leukemic Cells over time

// TODO: get WBCs and blasts from this data
// Ha + Hq
G := ktr;
Xtr' = Dh * Ha - G * Xtr;
Xwbc' = G * Xtr - kwbc * Xwbc;

// TODO: blasts don't self-reproduce. They should be a separate layer of differentiation from the leukemic stem cells?

//parameters
Tqh = 0.2;
Tql = 0.002;
Kq = 10000; // Confused on what this Parameter should be
Tah = 0.01;
Tal = 0.25; // Guestimate from Figure 2a
Ka = 10000; // Confused on what this Parameter should be
Pl = 0.099; // Guestimate from Figure 2a
Ph = 1/25;
Dh = 1/30;
Dl = 1/30;

// parameters for the WBC component
ktr = 0.236; // transition rate - unit: 1/day
gam = 0.651; // feedback of G-CSF on WBC proliferation - MAKE SURE THIS VARIABLE IS NAMED gam
B = 2; //4.67; // unit: G/L
B0 = 2; //4.67;
kwbc = 2.3765; 

// initial values
Xtr = B*kwbc/ktr;
Xwbc = B;

// kill factor
kf = 0.99;
kf_aza = 0.1;

// chemotherapy time periods
in_ven_treatment = 0;
in_aza_treatment = 0;
c := in_ven_treatment*kf + in_aza_treatment*kf_aza;



//initial values
Lq = 10000*.01; // Randomly Decided
Hq = 10000*.71; // Randomly Decided
La = 10000*.20; // Randomly Decided
Ha = 10000*.08; // Randomly Decided
La0 = La;
Lq0 = Lq;
Ha0 = Ha;
Hq0 = Hq;
//LkBd := log10(100*(La*100 + Lq*100)/(La + Ha + Lq + Hq));

Xblasts_obs := 100*(La + Lq)/(La + Ha + Lq + Hq);
"""

param_names_hoffmann_like = ['Tal', 'Pl', 'kf', 'kf_aza', 'ktr', 'B0', 'La0']
param_bounds_hoffmann_like = [(0,1), (0,1), (0,1.5), (0,1.5), (0,1), (0,10), (0, 9000)]

def initialization_fn_hoffmann_like(model, param_vals):
    # we assume a fixed initial ratio of quiescent to active cells for both the healthy and leukemic compartments
    # assume that quiescent leukemic cells are 5% of active leukemic cells, and a 9-1 ratio of quiescent to active healthy cells.
    La = param_vals[6]
    Lq = La*0.05
    model.setValue('Xwbc', param_vals[5])
    model.setValue('La', La)
    model.setValue('Lq', Lq)
    model.setValue('Hq', (10000 - (La+Lq))*0.9)
    model.setValue('Ha', (10000 - (La+Lq))*0.1)
    return model

def generate_dosing_component_hoffmann_like(venex_cycles, aza_cycles):
    """
    Params
    ------
        venex_cycles - pandas dataframe with columns start, end, dose
        aza_cycles - pandas dataframe with columns start, end, dose
    """
    output_str = ''
    for _, row in venex_cycles.iterrows():
        start = row.start
        end = row.end
        #dose = row.dose
        output_str += f'at time >= {start} : in_ven_treatment = 1;\n'
        output_str += f'at time >= {end} : in_ven_treatment = 0;\n'
    output_str += '\n'
    for _, row in aza_cycles.iterrows():
        start = row.start
        end = row.end
        #dose = row.dose
        # this represents the pulse of injections - 
        for day in range(start, end):
            output_str += f'at time >= {day} : in_aza_treatment = 1;\n'
            output_str += f'at time >= {day} + 0.2 : in_aza_treatment = 0;\n'
    return output_str



def build_pm_model_hoffmann_like(model, wbc, blasts, theta=None, n_samples=None, params_to_fit=None,
        initialization=None, use_b0=False, uniform_prior=False, **params):
    """
    Builds a PyMC model
    """
    if params_to_fit is None:
        params_to_fit = param_names_hoffmann_like
    if theta is None:
        model.resetToOrigin()
        default_params = [model.getValue(x) for x in params_to_fit]
        theta = default_params
    print('theta:', theta)
    pytensor_forward_function = generate_forward_function(model, wbc, blasts, n_samples=n_samples,
            params_to_fit=params_to_fit, initialization=initialization)
    wbc_data = wbc[1].to_numpy(dtype=np.float64)
    blast_data = blasts[1].to_numpy(dtype=np.float64)
    print('wbc_data:', wbc_data)
    print('blast_data:', blast_data)
    with pm.Model() as new_patient_model_pm:
        # priors
        # TODO: uniform priors for slope?
        if not uniform_prior:
            Tal = pm.TruncatedNormal("Tal", mu=theta[0], sigma=theta[0]/2, lower=0, upper=1, initval=theta[0])
            Pl = pm.TruncatedNormal("Pl", mu=theta[1], sigma=theta[1]/2, lower=0, upper=1, initval=theta[1])
            kf = pm.TruncatedNormal("kf", mu=theta[2], sigma=theta[2]/2, lower=0, initval=theta[2])
            kf_aza = pm.TruncatedNormal("kf_aza", mu=theta[3], sigma=theta[3]/2, lower=0, initval=theta[3])
            ktr = pm.TruncatedNormal("ktr", mu=theta[4], sigma=theta[4]/2, lower=0, initval=theta[4])
            B0 = pm.TruncatedNormal('B0', mu=theta[5], sigma=theta[5]/2, lower=0, initval=theta[5])
            La0 = pm.TruncatedNormal('La0', mu=theta[6], sigma=theta[6]/2, lower=0, initval=theta[6])
        else:
            Tal = pm.Uniform("Tal", lower=0, upper=1, initval=theta[0])
            Pl = pm.Uniform("Pl", lower=0, upper=1, initval=theta[1])
            kf = pm.Uniform("kf", lower=0, upper=2, initval=theta[2])
            kf_aza = pm.Uniform("kf_aza", lower=0, upper=2, initval=theta[3])
            ktr = pm.Uniform("ktr", lower=0, upper=2, initval=theta[4])
            B0 = pm.Uniform('B0', lower=0, upper=10, initval=theta[5])
            La0 = pm.Uniform('La0', lower=0, upper=10000, initval=theta[6])
        sigma_wbc = pm.HalfNormal("sigma_wbc", 5)
        sigma_blasts = pm.HalfNormal("sigma_blasts", 5)

        # ODE solution function
        ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, kf_aza, ktr, B0, La0]))
        if use_b0:
            ode_solution = pytensor_forward_function(pm.math.stack([Tal, Pl, kf, kf_aza, ktr, B0, La0]))
        
        # split up the blasts and WBCs
        wbc_ode = ode_solution[0,:len(wbc_data)]
        blast_ode = ode_solution[1,:len(blast_data)]

        # likelihood
        pm.Normal("Y_wbc_obs", mu=wbc_ode, sigma=sigma_wbc, observed=wbc_data)
        pm.Normal("Y_blast_obs", mu=blast_ode, sigma=sigma_blasts, observed=blast_data)
    return new_patient_model_pm

#####################################################################
# Hoffmann Model 2 - gamma-like feedback
