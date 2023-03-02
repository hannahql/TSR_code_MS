import numpy as np
import pandas as pd
import random

"""
Implements cluster randomized experiments in a market with two types of customers and 
two types of listings, with clustered preferences as defined in the paper. 
"""

def clustered_utilities(customer_types, listing_types, preference_dict, 
                        primary_util, secondary_util, treat_delta):
    """
    Defines utilities where customers prefer one type of listing, given in preference_dict.
    Have utility 'primary_util' for preferred listing, 'secondary_util' for all other listings
    Treatment induces multiplicative effect on all utilities, multiplying by 'treat_delta'.

    Outputs dictionary of utilities indexed by vs[customer_type][treatment_cond][listing_type]
    """
    vs = {}
    for c in customer_types:
        vs[c] = {}
        vs[c]['control'] = {}
        preferred_listing = preference_dict[c]
        vs[c]['control'][preferred_listing] = primary_util
        non_preferred_types = list(listing_types)
        non_preferred_types.remove(preferred_listing)
        for l in non_preferred_types:
            vs[c]['control'][l] = round(secondary_util,3)
            
        vs[c]['treat'] = {}
        for l in listing_types:
            vs[c]['treat'][l] = round(vs[c]['control'][l] * treat_delta,3)
            
    return vs


def cluster_by_listing_type(listing_types, a_L=0.5, completely_randomized=True):
    """
    Assigns one listing type to treatment and one to control.
    """
    treatment_assignments = {}
    if completely_randomized:
        n_treat = int(np.floor(len(listing_types)*a_L))
        treated_types = random.sample(listing_types, n_treat)
    for t in listing_types:
        if t in treated_types:
            treatment_assignments[t] = 'treat'
        else:
            treatment_assignments[t] = 'control'
    return treatment_assignments


def listing_cluster_exp_induced_parameters(listing_types, listing_type_assignments,
                                           rhos_pre_treat, cluster_a_L, 
                                        customer_types, alpha, vs, lam, customer_proportions):
    thetas_exp = []
    thetas_c = []
    thetas_t = []
    
    rhos_exp = {}
    rhos_c = {}
    rhos_t = {}
    
    for l in listing_types:
        if listing_type_assignments[l] == 'treat':
            thetas_t.append(str(l)+"_treat")
            rhos_t[str(l)+"_treat"] =  rhos_pre_treat[l] 
            thetas_exp.append(str(l)+"_treat")
            rhos_exp[str(l)+"_treat"] = rhos_pre_treat[l]
        else:
            thetas_c.append(str(l)+"_control")
            rhos_c[str(l)+"_control"] = rhos_pre_treat[l]
            thetas_exp.append(str(l)+"_control")
            rhos_exp[str(l)+"_control"] = rhos_pre_treat[l] 
            
    gammas_exp = customer_types
    alpha_gammas={gamma: alpha for gamma in gammas_exp}
    lam_gammas_exp={gamma:lam*customer_proportions[gamma] for gamma in gammas_exp}
    
    v_gammas_c = {}
    v_gammas_t = {}
    v_gammas_exp = {}
    for c in customer_types:
        v_gammas_c[c] = {str(k)+ "_control":v for k,v in vs[c]['control'].items()}
        v_gammas_t[c] = {str(k)+ "_treat":v for k,v in vs[c]['treat'].items()}
        v_gammas_exp[c] = {}
        for l in listing_type_assignments:
            assignment = listing_type_assignments[l]
            v_gammas_exp[c][str(l)+"_"+assignment] = vs[c][assignment][l]

    return {'thetas_exp':thetas_exp, 'thetas_c': thetas_c, 'thetas_t':thetas_t, 
             'rhos_exp': rhos_exp, 'rhos_c':rhos_c, 'rhos_t':rhos_t, 
             'gammas_exp':gammas_exp, 'alpha_gammas':alpha_gammas, 'lam_gammas_exp':lam_gammas_exp,
             'v_gammas_exp':v_gammas_exp, 'v_gammas_c':v_gammas_c, 'v_gammas_t':v_gammas_t}     


def calc_cluster_estimator_mean_field(booking_rates, thetas_c, thetas_t, cluster_a_L=0.5):
    treat_cluster_book_rate = 0
    control_cluster_book_rate = 0
    for c in booking_rates.keys(): #iterate through customers
        treat_cluster_book_rate += sum([booking_rates[c][l] for l in thetas_t])
        control_cluster_book_rate += sum([booking_rates[c][l] for l in thetas_c])
    treatment_est = treat_cluster_book_rate/cluster_a_L
    control_est = control_cluster_book_rate/(1-cluster_a_L)
    
    cluster_est = treatment_est - control_est
    
    return {'cluster_est':cluster_est, 'treatment_est':treatment_est, 'control_est':control_est}


def calc_cluster_estimate_finite(T_start, T_end, n, events, cluster_a_L, thetas_c, thetas_t):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end)]['choice_type']
    n_control = bookings.isin(thetas_c).sum()
    n_treat = bookings.isin(thetas_t).sum()
    
    estimate = (n_treat/(n*cluster_a_L) - n_control/(n*(1-cluster_a_L)))/(T_end-T_start)
    return estimate
    

def calc_cluster_estimates_from_events(events, exp_params, T_start, T_end,
                                       n_listings, cluster_a_L=0.5, 
                                       varying_time_horizons=False,
                                       save=False, file_path=None):
    start_times = T_start
    end_times = T_end
    
    estimates = {}
    for lam in events.keys():
        if varying_time_horizons:
            T_start = start_times[lam]
            T_end = end_times[lam]
        estimates[lam] = {}
        estimates[lam]['clustered'] = []
        for events_one_run in events[lam]['events']['clustered']:
            events_one_run = events_one_run[0]
            est = calc_cluster_estimate_finite(T_start, T_end, n_listings, events_one_run, cluster_a_L, 
                                            exp_params[lam]['thetas_c'], exp_params[lam]['thetas_t'])
            estimates[lam]['clustered'].append(est)
    estimates_df = {lam:estimates[lam]['clustered'] for lam in estimates.keys()}
    estimates_df = pd.DataFrame(estimates_df)
    if save:
        estimates_df.to_csv(file_path+"clustered_estimates.csv")
        
    return estimates_df
        
            