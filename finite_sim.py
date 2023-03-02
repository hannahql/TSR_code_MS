import numpy as np
import pandas as pd
#from numpy import random
import random
from operator import itemgetter
import copy

import exp_wrapper 
import experiment_helper
import pdb

from multiprocessing import Pool
from functools import partial


"""
Simulates a market with a finite number of listings, using 
continuous time Markov chain defined in the paper. 
Starts with all listings available at time t=0, and simulates
arrival of customers and resulting bookings over time. 

Parameters used across different functions:
s_t - dict current state
s_full - dict full availability
tau - float
lam_gammas - dict of arrival rates
Events - time series of state of system and events that happen.
Event_times - dictionary where keys are listings and values are the times
at which listing became booked or unavailable. 
Listing_dfs - df indexed by listing ID, gives number of bookings available at end of time horizon.
"""

def draw_next_event(n, s_t, s_full, thetas, gammas, tau, lam_gammas):
    """
    Simulates next event, whether listing replenishes first or another customer arrives first.
    """
    customer_arrival = False # whether customer arrives first
    next_listing = None
    next_customer = None
    elapsed_time = 0
    
    arrival_times = {}
    for gamma in gammas:
        arrival_times[gamma] = np.random.exponential(1/(n * lam_gammas[gamma]))
    first_customer = min(arrival_times.items(), key=itemgetter(1))
    
    if s_t==s_full:
        next_customer = first_customer[0]
        elapsed_time = first_customer[1]
        customer_arrival = True
    else: 
        rep_times = {} #replenishment
        for theta in thetas:
            num_booked = s_full[theta] - s_t[theta]
            if num_booked > 0:
                rep_times[theta] = min(np.random.exponential(1/tau, num_booked))
        first_listing = min(rep_times.items(), key=itemgetter(1))

        if first_listing[1] < first_customer[1]: #listing replenishes first
            next_listing = first_listing[0]
            elapsed_time = first_listing[1]
        else:
            next_customer = first_customer[0]
            elapsed_time = first_customer[1]
            customer_arrival = True
        
    return {'customer_arrival':customer_arrival, 'next_listing':next_listing,
           'next_customer':next_customer, 'elapsed_time':elapsed_time}


def customer_choice_alpha(n, s_t, thetas, gamma, v_gamma, alpha, epsilon):
    """
    Upon arrival of a customer, customer samples items into their consideration set
    with probability alpha.
    """
    n_total_avail = sum(s_t.values())
    sampling_probs = {theta: s_t[theta] for theta in thetas}
    norm_constant = sum(sampling_probs.values())
    sampling_probs = {theta: sampling_probs[theta]/norm_constant for theta in thetas}
    
    all_available =  [[theta]*int(sampling_probs[theta]*n) for theta in thetas]
    all_available = [item for sublist in all_available for item in sublist]
    choice_set = random.sample(all_available, int(n_total_avail*alpha))
    
    choices, counts = np.unique(choice_set, return_counts=True)
    choice_counts = dict(zip(choices, counts))
    
    #choose from choice set
    choice_set_util = sum([v_gamma[choice]*choice_counts[choice] 
                           for choice in choice_counts.keys()])

    choice_probs = {choice:choice_counts[choice]*v_gamma[choice]/(choice_set_util+epsilon*n) 
                    for choice in choice_counts.keys()}
    choice_probs['outside_option'] = epsilon*n / (choice_set_util + epsilon*n)

    choices, probs = zip(*choice_probs.items())
    choice = np.random.choice(choices, p=probs)
    
    return choice


def customer_choice_finite_k_no_rec(n, k, s_t, thetas, gamma, v_gamma, alpha, epsilon):
    """
    Modified consideration set formation, as described in Appendix,
    where customer draws a fixed number k from the set of available listings
    If there are fewer than k listings available, customer samples all available listings.
    """
    sampling_probs = {theta: s_t[theta] for theta in thetas}
    norm_constant = sum(sampling_probs.values())
    sampling_probs = {theta: sampling_probs[theta]/norm_constant for theta in thetas}
    
    all_available =  [[theta]*int(sampling_probs[theta]*n) for theta in thetas]
    all_available = [item for sublist in all_available for item in sublist]
    choice_set = random.sample(all_available, min(k,len(all_available)))

    
    choices, counts = np.unique(choice_set, return_counts=True)
    choice_counts = dict(zip(choices, counts))
    
    choice_set_util = sum([v_gamma[choice]*choice_counts[choice] 
                           for choice in choice_counts.keys()])

    choice_probs = {choice:choice_counts[choice]*v_gamma[choice]/(choice_set_util+epsilon*n) 
                    for choice in choice_counts.keys()}
    choice_probs['outside_option'] = epsilon*n / (choice_set_util + epsilon*n)

    choices, probs = zip(*choice_probs.items())
    choice = np.random.choice(choices, p=probs)
    
    return choice



def customer_choice(choice_set_type, n, k, s_t, thetas, gamma, vs_gamma, alpha, epsilon):
    if choice_set_type == 'alpha':
        choice = customer_choice_alpha(n, s_t, thetas, gamma, vs_gamma, alpha, epsilon)
    elif choice_set_type == 'finite_k':
        choice = customer_choice_finite_k_no_rec(n, k, s_t, thetas, gamma, vs_gamma, alpha, epsilon)
    else:
        return "Invalid Choice Type."
    return choice



def run_mc_listing_ids(choice_set_type, n, k, s_0, s_full, T, thetas, gammas, vs, 
                        tau, lam_gammas, alpha, epsilon, run_number):
    """
    Runs the continuous time Markov Chain. 
    State refers to the state of the market just before customer arrives or listing replenishes.
    """
    np.random.seed()
    
    t = 0
    s_t = copy.copy(s_0)
    
    times = []
    states = []

    is_replenished = []
    is_customer = []
    customer_type = []
    choice_type = []
    
    # Initialize listings_df - tracks each listing and its type 
    # and availability status
    n_listings = sum(s_full.values())
    listing_types = []
    for l in s_0.keys():
        listing_types.extend([l]*s_0[l])
    availability = [1]*n_listings
    listing_ids = ["l_"+str(i) for i in range(n_listings)]
    listings_df = pd.DataFrame({'type':listing_types, 'available':availability}, 
                               index=listing_ids)
    
    # Initialize listing_times 
    # Records the times that each listing has its 
    # availability status changed
    listing_times = {}
    for l in listing_ids:
        listing_times[l] = []
    
    while t<T:
        states.append(copy.copy(s_t))
        times.append(t)
        # next event returns customer type if event is customer arrival 
        # and listing type if event is listing being replenished
        next_event = draw_next_event(n, s_t, copy.copy(s_full), 
                                     thetas, gammas, tau, lam_gammas)
        if next_event['customer_arrival']:
            if sum(s_t.values()) ==0:
                choice = 'outside_option'
            else: 
                gamma = next_event['next_customer']
                
                choice = customer_choice(choice_set_type, n, k, s_t, 
                                         thetas, gamma, vs[gamma], 
                                         alpha, epsilon)
                if choice != 'outside_option':
                    s_t[choice] -= 1 
                    # randomly select an available listing of the 
                    # appropriate type to be chosen. Update listings_df 
                    # and listing_times
                    available_set = listings_df[(listings_df['type']==choice)
                                                & (listings_df['available']==1)]
                    the_chosen_one = available_set.sample(n=1).index[0]
                    listings_df.loc[the_chosen_one, 'available'] = 0
                    listing_times[the_chosen_one].append(t)
            
            # updates total list of events
            is_replenished.append(0)
            is_customer.append(1)
            customer_type.append(gamma)
            choice_type.append(choice)
        else: # next event is a listing being replenished
            # update state of MC
            l = next_event['next_listing']
            s_t[l] += 1
            
            # Update listings_df. Randomly select an unavailable listing 
            # of the appropriate type to be replenished
            unavailable_set = listings_df[(listings_df['type']==l)
                                            & (listings_df['available']==0)]
            the_chosen_one = unavailable_set.sample(n=1).index[0]
            listings_df.loc[the_chosen_one, 'available'] = 1
            listing_times[the_chosen_one].append(t)
            
            is_replenished.append(1)
            is_customer.append(0)
            customer_type.append(None)
            choice_type.append(None)
        t += next_event['elapsed_time']
    
    events = pd.DataFrame(data=states)
    events['time'] = times
    events['is_replenished'] = is_replenished
    events['is_customer'] = is_customer
    events['customer_type'] = customer_type
    events['choice_type'] = choice_type

    return {'events':events, 'listing_times':listing_times, 'listing_dfs':listings_df}
        
    
def calc_lr_estimate(T_start, T_end, n, events, a_L, thetas_c, thetas_t):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end)]['choice_type']
    n_control = bookings.isin(thetas_c).sum()
    n_treat = bookings.isin(thetas_t).sum()

    estimate = (n_treat/(n*a_L) - n_control/(n*(1-a_L)))/(T_end-T_start)
    return estimate


def calc_cr_estimate(T_start, T_end, n, events, a_C, gammas_c, gammas_t):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end) & (events['choice_type']!='outside_option')]
    n_control_bookings = bookings['customer_type'].isin(gammas_c).sum()
    n_treat_bookings = bookings['customer_type'].isin(gammas_t).sum()
    
    estimate = (n_treat_bookings/(n*a_C)
                - n_control_bookings/(n*(1-a_C)))/ (T_end-T_start)

    return estimate


def calc_global_estimate(T_start, T_end, n, events, gammas):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end) & (events['choice_type']!='outside_option')]
    n_bookings = len(bookings)
    
    global_est = n_bookings / (n * (T_end-T_start))
    return global_est


def calc_tsr_estimate(T_start, T_end, n, customer_side_weight, events, a_C, a_L, 
                      gammas_c, gammas_t, thetas_c, thetas_t, tsr_est_types):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end) & (events['choice_type']!='outside_option')]

    bookings_00 = bookings[bookings['customer_type'].isin(gammas_c) 
                           & bookings['choice_type'].isin(thetas_c)]
    bookings_01 = bookings[bookings['customer_type'].isin(gammas_c) 
                           & bookings['choice_type'].isin(thetas_t)]
    bookings_10 = bookings[bookings['customer_type'].isin(gammas_t) 
                           & bookings['choice_type'].isin(thetas_c)]
    bookings_11 = bookings[bookings['customer_type'].isin(gammas_t) 
                           & bookings['choice_type'].isin(thetas_t)]
    
    n_00 = len(bookings_00)
    n_01 = len(bookings_01)
    n_10 = len(bookings_10)
    n_11 = len(bookings_11)
    
    
    scale = customer_side_weight*(1-customer_side_weight)
    
    estimators = {}
    
    tsr_est_naive = (n_11/(a_C*a_L) - (n_01 + n_10 + n_00)/(1-a_C*a_L))/(n*(T_end-T_start))
    
    estimators['tsr_est_naive'] = tsr_est_naive
    
    tsris = [est for est in tsr_est_types if "tsri" in est]
    k_values = [float(tsri[len(("tsri_")):]) for tsri in tsris]
    for k in k_values:
        tsri_est = (customer_side_weight*(n_11/(a_C*a_L) - n_01/((1-a_C)*a_L)) 
                 + (1-customer_side_weight)* (n_11/(a_C*a_L) - n_10/((a_C)*(1-a_L)))
                  - 2*k*scale * n_00/((1-a_C)*(1-a_L))
                  + k*scale * n_01/((1-a_C)*a_L)
                  + k*scale * n_10/(a_C * (1-a_L))
                 )/(n*(T_end-T_start))
        estimators['tsri_'+str(k)] = tsri_est

    return estimators


    
def calc_global_cond_rates(T_start, T_end, n, events):
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end) & (events['choice_type']!='outside_option')]
    n_bookings = len(bookings)
    booking_rate = n_bookings / n / (T_end - T_start)

    return booking_rate
    
    
def calc_bookings_per_cell(T_start, T_end, events, 
                      gammas_c, gammas_t, thetas_c, thetas_t):
    
    bookings = events[(events['is_customer']==1) & (events['time']>T_start)
                     & (events['time']<T_end) & (events['choice_type']!='outside_option')]

    bookings_00 = bookings[bookings['customer_type'].isin(gammas_c) 
                           & bookings['choice_type'].isin(thetas_c)]
    bookings_01 = bookings[bookings['customer_type'].isin(gammas_c) 
                           & bookings['choice_type'].isin(thetas_t)]
    bookings_10 = bookings[bookings['customer_type'].isin(gammas_t) 
                           & bookings['choice_type'].isin(thetas_c)]
    bookings_11 = bookings[bookings['customer_type'].isin(gammas_t) 
                           & bookings['choice_type'].isin(thetas_t)]
    
    n_00 = len(bookings_00)
    n_01 = len(bookings_01)
    n_10 = len(bookings_10)
    n_11 = len(bookings_11)
    total = n_00 + n_01 + n_10 + n_11
    return{'n_00':n_00, 'n_01':n_01, 'n_10':n_10, 'n_11':n_11, 'total':total}
    
    


            

    

        
