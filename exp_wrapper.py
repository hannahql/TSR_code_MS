import numpy as np

import experiment_helper as experiment
import cluster_rand as cluster


"""
Wrapper function around experiment_helper to input model parameters 
from the global control setting, along with changes in utilities induced by treatment,
to define the resulting parameters for each of the experiments.

Parameters of the model, used across different functions. 

state: dictionary representing state of the Markov chain, for each 
        listing type denotes mass available
thetas: list of all listing types
gammas: list of all customer types
v_gammas_c: dictionary of the control utilities
v_gammas_t: dictionary of the treatment utilities
epsilon: float representing outside option
alpha: float representing probability of inclusion in choice set
choice_model: string, implements logit choice model

"""

def run_experiment(experiment_type, epsilon, tau, 
                   thetas_exp, rhos_exp, rhos_c, rhos_t, 
                   gammas_exp, 
                   alpha_gammas, lam_gammas_exp, 
                   v_gammas_exp, v_gammas_c, v_gammas_t,
                   lam_gammas_c=None, lam_gammas_t=None,
                   thetas_c=None, thetas_t=None,
                   gammas_c=None, gammas_t=None,
                   a_C=None, a_L=None, 
                   cluster_a_L=None,
                   cluster_assignments=None,
                   customer_side_weight=None,
                   scale=None):
    
    #solve for optimal solution
    sol = experiment.minimize_w(thetas=thetas_exp, gammas=gammas_exp, rhos=rhos_exp, 
                     lam_gammas=lam_gammas_exp,
                     tau=tau, epsilon=epsilon, 
                     alpha_gammas=alpha_gammas, v_gammas=v_gammas_exp)
    # convert to states s
    solution = {key: np.exp(value) for key, value in sol.items()}

    booking_probs = experiment.calc_all_booking_probs(solution, thetas=thetas_exp, 
                                               gammas=gammas_exp, v_gammas=v_gammas_exp,
                                               epsilon=epsilon, alpha_gammas=alpha_gammas)
    booking_rates = experiment.calc_all_booking_rates(booking_probs, lam_gammas_exp)
    

    if experiment_type=="listing":
        est = experiment.calc_naive_estimator('listing', booking_rates, a_L=a_L,
                                                thetas_c=thetas_c, thetas_t=thetas_t)
    elif experiment_type=="customer":
        est = experiment.calc_naive_estimator('customer', booking_rates, a_C=a_C,
                                        gammas_c=gammas_c, gammas_t=gammas_t)
    elif experiment_type=="tsr":
        est = experiment.calc_tsr_estimator(booking_rates, 
                                            thetas_c, thetas_t,
                                            gammas_c, gammas_t,
                                            a_C=a_C, a_L=a_L, 
                                            customer_side_weight=customer_side_weight,
                                            scale=scale)
    elif experiment_type=='clustered':
        est = cluster.calc_cluster_estimator(booking_rates, thetas_c, thetas_t)
    
    return {'estimator':est, 'solution':solution}



def listing_side_exp_induced_parameters(listing_types, rhos_pre_treat, a_L, 
                                     customer_types, alpha, vs, lam, customer_proportions):
    """
    Given model parameters for global control setting, returns the induced parameters 
    for and experiment setting.
    """
    thetas_exp = []
    thetas_c = []
    thetas_t = []
    for l in listing_types:
        thetas_exp.append(str(l)+"_control")
        thetas_exp.append(str(l)+"_treat")

        thetas_c.append(str(l)+"_control")
        thetas_t.append(str(l)+"_treat")

    rhos_exp = {}
    rhos_c = {}
    rhos_t = {}
    for l in listing_types:
        rhos_exp[str(l)+"_control"] = (1-a_L) * rhos_pre_treat[l]
        rhos_exp[str(l)+"_treat"] = a_L * rhos_pre_treat[l]

        rhos_c[str(l)+"_control"] = rhos_pre_treat[l]
        rhos_t[str(l)+"_treat"] =  rhos_pre_treat[l]

    gammas_exp = customer_types
    alpha_gammas={gamma: alpha for gamma in gammas_exp}
    lam_gammas_exp={gamma:lam*customer_proportions[gamma] for gamma in gammas_exp}

    listing_vs_control = {}
    listing_vs_treat = {}
    for g in customer_types:   
        listing_vs_control[g] = {str(k)+ "_control":v for k,v in vs[g]['control'].items()}
        listing_vs_treat[g] = {str(k)+ "_treat":v for k,v in vs[g]['treat'].items()}

    v_gammas_exp = {gamma:{**listing_vs_control[gamma], **listing_vs_treat[gamma]} for gamma in gammas_exp}
    v_gammas_c = listing_vs_control
    v_gammas_t = listing_vs_treat
    return {'thetas_exp':thetas_exp, 'thetas_c': thetas_c, 'thetas_t':thetas_t, 
             'rhos_exp': rhos_exp, 'rhos_c':rhos_c, 'rhos_t':rhos_t, 
             'gammas_exp':gammas_exp, 'alpha_gammas':alpha_gammas, 'lam_gammas_exp':lam_gammas_exp,
             'v_gammas_exp':v_gammas_exp, 'v_gammas_c':v_gammas_c, 'v_gammas_t':v_gammas_t}

def customer_side_exp_induced_parameters(listing_types, rhos_pre_treat, a_C, 
                                      customer_types, alpha, vs, lam, customer_proportions):
    """
    Given model parameters for global control setting, returns the induced parameters 
    for and experiment setting.
    """
    thetas_exp = listing_types
    rhos_exp = rhos_pre_treat
    rhos_c = rhos_pre_treat
    rhos_t = rhos_pre_treat

    v_gammas_exp = {}
    for g in customer_types:
        v_gammas_exp[str(g)+"_control"] = vs[g]["control"]
        v_gammas_exp[str(g)+"_treat"] = vs[g]["treat"]
    v_gammas_c = {str(key)+"_control":val['control'] for key,val in vs.items()}
    v_gammas_t = {str(key)+"_treat":val['treat'] for key,val in vs.items()}

    gammas_exp = []
    gammas_c = []
    gammas_t = []
    for g in customer_types:
        gammas_exp.append(str(g)+"_control")
        gammas_exp.append(str(g)+"_treat")

        gammas_c.append(str(g)+"_control")
        gammas_t.append(str(g)+"_treat")
    alpha_gammas={gamma: alpha for gamma in gammas_exp}
    lam_gammas_c = {str(gamma)+"_control":lam*customer_proportions[gamma] for gamma in customer_types}
    lam_gammas_t = {str(gamma)+"_treat":lam*customer_proportions[gamma] for gamma in customer_types}

    lam_gammas_exp = {}
    for g in customer_types:
        lam_gammas_exp[str(g)+"_control"] = lam*customer_proportions[g]*(1-a_C)
        lam_gammas_exp[str(g)+"_treat"] = lam*customer_proportions[g]*a_C
    
    return {'thetas_exp':thetas_exp, 
             'rhos_exp': rhos_exp, 'rhos_c':rhos_c, 'rhos_t':rhos_t, 
             'gammas_exp':gammas_exp, 'gammas_c':gammas_c, 'gammas_t':gammas_t,
            'alpha_gammas':alpha_gammas, 
            'lam_gammas_exp':lam_gammas_exp, 'lam_gammas_c':lam_gammas_c, 'lam_gammas_t':lam_gammas_t,
             'v_gammas_exp':v_gammas_exp, 'v_gammas_c':v_gammas_c, 'v_gammas_t':v_gammas_t}

def tsr_exp_induced_parameters(listing_types, rhos_pre_treat, a_C, a_L, 
                              customer_types, customer_proportions, alpha, vs, lam):
    thetas_exp = []
    thetas_c = []
    thetas_t = []
    for l in listing_types:
        thetas_exp.append(str(l)+"_control")
        thetas_exp.append(str(l)+"_treat")

        thetas_c.append(str(l)+"_control")
        thetas_t.append(str(l)+"_treat")

    gammas_exp = []
    gammas_c = []
    gammas_t = []
    for g in customer_types:
        gammas_exp.append(str(g)+"_control")
        gammas_exp.append(str(g)+"_treat")

        gammas_c.append(str(g)+"_control")
        gammas_t.append(str(g)+"_treat")

    rhos_exp = {}
    rhos_c = {}
    rhos_t = {}
    for l in listing_types:
        rhos_exp[str(l)+"_control"] = (1-a_L) * rhos_pre_treat[l]
        rhos_exp[str(l)+"_treat"] = a_L * rhos_pre_treat[l]

        rhos_c[str(l)+"_control"] = rhos_pre_treat[l]
        rhos_t[str(l)+"_treat"] =  rhos_pre_treat[l]

    v_gammas_exp = {}
    for g in customer_types: #utilities for control customers
        v_gammas_exp[str(g)+'_control'] = {}
        for l in listing_types: 
            v_gammas_exp[str(g)+'_control'][str(l)+"_control"] = vs[g]['control'][l]
            v_gammas_exp[str(g)+'_control'][str(l)+"_treat"] = vs[g]['control'][l]
    for g in customer_types: #utilities for treatment customers
        v_gammas_exp[str(g)+'_treat'] = {}
        for l in listing_types:
            v_gammas_exp[str(g)+'_treat'][str(l)+"_control"] = vs[g]['control'][l]
            v_gammas_exp[str(g)+'_treat'][str(l)+"_treat"] = vs[g]['treat'][l]

    v_gammas_c = {}
    v_gammas_t = {}
    for g in customer_types:   
        v_gammas_c[str(g)+"_control"] = {str(k)+ "_control":v for k,v in vs[g]['control'].items()}
        v_gammas_t[str(g)+"_treat"] = {str(k)+ "_treat":v for k,v in vs[g]['treat'].items()}

    alpha_gammas={gamma: alpha for gamma in gammas_exp}
    lam_gammas_c = {str(gamma)+"_control":lam*customer_proportions[gamma] for gamma in customer_types}
    lam_gammas_t = {str(gamma)+"_treat":lam*customer_proportions[gamma] for gamma in customer_types}
    lam_gammas_exp = {} 
    for g in customer_types:
        lam_gammas_exp[str(g)+"_control"] = lam*customer_proportions[g]*(1-a_C)
        lam_gammas_exp[str(g)+"_treat"] = lam*customer_proportions[g]*a_C
    
    return {'thetas_exp':thetas_exp, 'thetas_c': thetas_c, 'thetas_t':thetas_t, 
            'rhos_exp': rhos_exp, 'rhos_c':rhos_c, 'rhos_t':rhos_t, 
            'gammas_exp':gammas_exp, 'gammas_c':gammas_c, 'gammas_t':gammas_t,
            'alpha_gammas':alpha_gammas, 
        'lam_gammas_exp':lam_gammas_exp, 'lam_gammas_c':lam_gammas_c, 'lam_gammas_t':lam_gammas_t,
            'v_gammas_exp':v_gammas_exp, 'v_gammas_c':v_gammas_c, 'v_gammas_t':v_gammas_t}
    
    
def calc_gte_from_exp_type(experiment_type, tau, epsilon,
                           thetas_exp, rhos_exp, rhos_c, rhos_t, 
                           gammas_exp, alpha_gammas, 
                           lam_gammas_exp, 
                           v_gammas_exp, v_gammas_c, v_gammas_t,
                           lam_gammas_c=None, lam_gammas_t=None,
                           thetas_c=None, thetas_t=None,
                           gammas_c=None, gammas_t=None,
                           a_L=None, a_C=None):
    if experiment_type=="listing":
        gte = experiment.calc_gte(thetas_c, thetas_t, gammas_exp, gammas_exp, 
                                 rhos_c, rhos_t,
                                 lam_gammas_exp, tau, epsilon, alpha_gammas, v_gammas_c, v_gammas_t)
    elif experiment_type=="customer":
        gte = experiment.calc_gte(thetas_exp, thetas_exp, gammas_c, gammas_t, 
                                         rhos_c, rhos_t,
                                         {**lam_gammas_c, **lam_gammas_t}, tau, epsilon, alpha_gammas, 
                                          v_gammas_c, v_gammas_t)
    elif experiment_type=="tsr":
        gte = experiment.calc_gte(thetas_c, thetas_t, gammas_c, gammas_t, 
                                     rhos_c, rhos_t,
                                     {**lam_gammas_c, **lam_gammas_t}, tau, epsilon, alpha_gammas, 
                                      v_gammas_c, v_gammas_t)
    return gte


def avg_gc_gt_booking_probs(listing_types, rhos_pre_treat, customer_types, customer_proportions, 
                              vs, alpha, epsilon, lam, tau):
    tsr_param = tsr_exp_induced_parameters(listing_types, rhos_pre_treat, 
                                                        .5, .5, 
                                                        customer_types, customer_proportions, 
                                                        alpha, vs, lam)
    global_solution = calc_gte_from_exp_type("tsr", tau, epsilon, **tsr_param) 
    
    gc_booking_probs = {customer: sum(global_solution['booking_probs_gc'][customer].values()) 
                        for customer in tsr_param['gammas_c']}
    gc_avg_booking_prob = np.mean(list(gc_booking_probs.values()))
    
    gt_booking_probs = {customer: sum(global_solution['booking_probs_gt'][customer].values()) 
                        for customer in tsr_param['gammas_t']}
    gt_avg_booking_prob = np.mean(list(gt_booking_probs.values()))
    
    return {'gc_avg_booking_prob': gc_avg_booking_prob, 'gt_avg_booking_prob':gt_avg_booking_prob}

    