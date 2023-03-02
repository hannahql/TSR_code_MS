import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize
from functools import partial

"""
Implements numerics for the mean field system. Implements the
optimization problem defined in Theorem 1 to solve for the mean field steady state,
then uses booking rates in the steady state to calculate GTE and estimates. 

Parameters of the model, used across different functions. 

state: dictionary representing state of the Markov chain, for each 
        listing type denotes mass available
thetas: list of all listing types
gammas: list of all customer types
v_gammas: dictionary of the utility that a type gamma customer has for a type theta listing
epsilon: float representing outside option
alpha: float representing probability of inclusion in choice set
choice_model: string, implements logit choice model


"""


def calc_all_booking_probs(state, thetas, gammas, v_gammas, epsilon, alpha_gammas):
    """
    Given the state of the system and model parameters, returns the booking probabilities
    that a customer of type gamma has for booking a listing of type theta
    
    """
    choice_probs = {}
    for gamma in gammas:
        choice_probs[gamma] = {}
        denom = (epsilon 
                    + alpha_gammas[gamma]*sum([state[theta]*v_gammas[gamma][theta] for theta in thetas])
                    )
        for theta in thetas:
            choice_probs[gamma][theta] = alpha_gammas[gamma]*state[theta]*v_gammas[gamma][theta] / denom
    return choice_probs


def calc_all_booking_rates(booking_probs, lam_gammas):
    """
    Given booking probabilities and arrival rates of customers lam_gammas,
    returns dictionary of booking rates.
    """
    booking_rates = {}
    for gamma in booking_probs.keys():
        booking_rates[gamma] = {}
        for theta in booking_probs[gamma].keys():
            booking_rates[gamma][theta] = lam_gammas[gamma]*booking_probs[gamma][theta]
            
    return booking_rates


def calc_naive_estimator(experiment_type, booking_rates, 
                         a_L=None, a_C=None, 
                         thetas_c = None, thetas_t=None, 
                        gammas_c = None, gammas_t=None):
    """
    Calculates naive estimator from an experiment type.
    Experiment type can be 'listing' (LR) or 'customer' (CR).
    """
    if experiment_type=='listing':
        treat_listing_book_rate = 0
        control_listing_book_rate = 0
        for gamma in booking_rates.keys():
            treat_listing_book_rate += sum([booking_rates[gamma][theta] for theta in thetas_t])
            control_listing_book_rate += sum([booking_rates[gamma][theta] for theta in thetas_c])

        naive_est = treat_listing_book_rate/a_L -  control_listing_book_rate/(1-a_L)
            
        treatment_est = treat_listing_book_rate/a_L
        control_est = control_listing_book_rate/(1-a_L)
    
    elif experiment_type=='customer':
        control_customer_book_rate = sum([ sum(booking_rates[gamma].values()) for gamma in gammas_c ])
        treat_customer_book_rate = sum([ sum(booking_rates[gamma].values()) for gamma in gammas_t ])
        naive_est = treat_customer_book_rate/a_C - control_customer_book_rate /(1-a_C)
        treatment_est = treat_customer_book_rate/a_C
        control_est = control_customer_book_rate/(1-a_C)
    return {'naive_est':naive_est, 'treatment_est':treatment_est, 'control_est':control_est}



def calc_tsr_estimator(booking_rates, thetas_c, thetas_t, 
                      gammas_c, gammas_t, a_C=None, a_L=None, 
                      customer_side_weight=None, scale=None):
    # In the 2x2 grid of the TSR design, 
    # calculate booking rates in each of the cells.

    cc = sum([booking_rates[c][l] for l in thetas_c for c in gammas_c])
    ct = sum([booking_rates[c][l] for l in thetas_t for c in gammas_c])
    tc = sum([booking_rates[c][l] for l in thetas_c for c in gammas_t])
    tt = sum([booking_rates[c][l] for l in thetas_t for c in gammas_t])


    # Calculates estimates of customer competition and listing competition,
    # as defined in the paper. 
    customer_comp = cc/((1-a_C)*(1-a_L)) - ct/((1-a_C)*a_L) 
    listing_comp = cc/((1-a_C)*(1-a_L)) - tc/(a_C*(1-a_L)) 
    
    
    #TSR-Naive estimator
    tsr_est_naive = tt / (a_C * a_L) - (ct + tc + cc)/ (1 - a_C * a_L)
    

    # Implements TSRI-1 and TSRI-2
    tsri_1 = (customer_side_weight * (tt/(a_C*a_L) - ct/((1-a_C)*a_L ))  #customer side est
                    + (1-customer_side_weight) *(tt/(a_C*a_L) - tc/(a_C*(1-a_L))) #listing side est
                    - 2 * scale * cc/((1-a_C)*(1-a_L))
                    + 1 * scale * ct/((1-a_C)*a_L)
                    + 1 * scale * tc/(a_C*(1-a_L))
                )

    tsri_2 = (customer_side_weight * (tt/(a_C*a_L) - ct/((1-a_C)*a_L ))  #customer side est
                    + (1-customer_side_weight) *(tt/(a_C*a_L) - tc/(a_C*(1-a_L))) #listing side est
                    - 4 * scale * cc/((1-a_C)*(1-a_L))
                    + 2 * scale * ct/((1-a_C)*a_L)
                    + 2 * scale * tc/(a_C*(1-a_L))
                )
    
    
   
    return {'cc': cc, 'ct': ct, 'tc': tc, "tt":tt,
            'tsr_est_naive':tsr_est_naive,
            'tsri_1.0':tsri_1, 'tsri_2.0':tsri_2,
            'customer_comp':customer_comp, 'listing_comp':listing_comp}
    



def calc_treat_control_booking_rates(state, thetas_c, thetas_t, gammas, v_gammas, 
                                epsilon, alpha_gammas, lam_gammas):
    """
    Given state of system and model parameters, calculates the treatment
    and control booking rates.
    """
    rate_treat_listing_booked = 0
    rate_control_listing_booked = 0

    for gamma in gammas:
        denom_gamma = (epsilon + alpha_gammas[gamma]*sum([state[theta]*v_gammas[gamma][theta] for theta in thetas_c]) 
                       + alpha_gammas[gamma]*sum([state[theta] * v_gammas[gamma][theta] for theta in thetas_t])
                       )
        
        rate_treat_listing_booked += (lam_gammas[gamma] * alpha_gammas[gamma]*sum([state[theta]*v_gammas[gamma][theta] for theta in thetas_t])
                                      / denom_gamma)
        rate_control_listing_booked += (lam_gammas[gamma] * alpha_gammas[gamma]*sum([state[theta]*v_gammas[gamma][theta] for theta in thetas_c])
                                        / denom_gamma)

    booking_rates = {'rate_treat_listing_booked':rate_treat_listing_booked, 
                     'rate_control_listing_booked':rate_control_listing_booked, 
                     'naive_estimator':rate_treat_listing_booked/rate_control_listing_booked}
    return booking_rates




def calc_gte(thetas_c, thetas_t, gammas_c, gammas_t, 
               rhos_c, rhos_t,
             lam_gammas, tau, epsilon, alpha_gammas, v_gammas_c, v_gammas_t):
    
    sol_y_gc = minimize_w(thetas_c, gammas_c, rhos_c, lam_gammas, tau, epsilon, alpha_gammas, v_gammas_c)
    sol_s_gc = {key:np.exp(value) for key, value in sol_y_gc.items()}

    sol_y_gt = minimize_w(thetas_t, gammas_t, rhos_t, lam_gammas, tau, epsilon, alpha_gammas, v_gammas_t)
    sol_s_gt = {key:np.exp(value) for key, value in sol_y_gt.items()}

    booking_probs_gc = calc_all_booking_probs(sol_s_gc, thetas_c, gammas_c, v_gammas_c, epsilon, 
                                              alpha_gammas)
    booking_rates_gc = calc_all_booking_rates(booking_probs_gc, lam_gammas)
    
    booking_probs_gt = calc_all_booking_probs(sol_s_gt, thetas_t, gammas_t, v_gammas_t, epsilon, 
                                              alpha_gammas)
    booking_rates_gt = calc_all_booking_rates(booking_probs_gt, lam_gammas)


    gte = (sum([sum(booking_rates_gt[gamma].values()) for gamma in gammas_t])
            - sum([sum(booking_rates_gc[gamma].values()) for gamma in gammas_c])
            )
    return {'gte':gte, 'solution_gc':sol_s_gc, 'solution_gt':sol_s_gt, 
            'booking_probs_gc':booking_probs_gc, 'booking_probs_gt':booking_probs_gt}



def function_w(y, thetas, gammas, rhos, lam_gammas, tau, epsilon, alpha_gammas, v_gammas):
    """
    Defines the objective function in Theorem 1 that is used to 
    find the steady state of the mean field system. 
    """
    v_y = (sum([lam_gammas[gamma] 
                 * np.log(epsilon + alpha_gammas[gamma] 
                          * np.sum(np.multiply(np.exp(y),v_gammas[gamma])) 
                        )
                 for gamma in gammas])
             - tau * np.sum(np.multiply(rhos, y))
             + tau * np.sum(np.exp(y))
            )
    return v_y



def minimize_w(thetas, gammas, rhos, lam_gammas, tau, epsilon, alpha_gammas, v_gammas):
    """
    Solves for the steady state of the mean field system
    by minimizing the objective function W(s) defined in Theorem 1,
    subject to feasibility constraints.
    """
    s_thetas = sorted(thetas)
    s_rhos = [item[1] for item in sorted(rhos.items())]
    
    s_gammas = sorted(gammas)
    s_v_gammas = {}
    for gamma in gammas:
        s_v_gammas[gamma] = [item[1] for item in sorted(v_gammas[gamma].items())]
    
    num_types = len(thetas)
    bounds = Bounds(np.repeat(-np.inf, num_types), s_rhos)
    y0 = np.repeat(-1, num_types)
    solve_v = minimize(partial(function_w, thetas=s_thetas, gammas=s_gammas, rhos=s_rhos, 
                               lam_gammas=lam_gammas, tau=tau, epsilon=epsilon, 
                               alpha_gammas=alpha_gammas, v_gammas=s_v_gammas),
                       y0,
                      method='SLSQP', 
                      options={ 'ftol': 1e-10},
                      bounds=bounds)

    sol = dict(zip(s_thetas, solve_v.x))
    return sol
    




