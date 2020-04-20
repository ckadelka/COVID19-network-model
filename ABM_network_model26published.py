#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:19:45 2020

@author: ckadelka
"""

'''interesting papers:
Displaying community structure of small-world networks
https://www.sciencedirect.com/science/article/pii/S1568494616300242

https://www.livescience.com/coronavirus-spread-after-recovery.html
'''

'''
00: basic SEIHRD model (H=hospitalized infected, D=dead)
01: added limited hospital size  (limited number of beds/respirators/ICUs)
    by adding a group C of ppl who require hospitalization/care but cannot get it
02: added social distancing between non-vulnerable and vulnerable population
03: added daily shopping trips/group gatherings with random people
04: changes: distributions for transition times, rather than Markov chain-induced geometric distribution,
    at each transition, each individual gets assigned a random time until the next transition
05: when ppl become hospitalized, they have a randomly assigned time until recovery (severity of infection/implications)
    any hospitalized has the same per-day death rate
    when hospitalized don't receive perfect care, the recovery takes longer (each day in the hospital they move <1 to recovery)
06: Shortened parameter names, deleted obsolete code, incluced watts_strogatz_graph networkx-independent, added estimation of R0
07: skipped
08: skipped
09: model implemented as a function, basic plotting added
10: stripped version for parallel computing, added SLURM_ID support, runs different social distancing choices (not same R0 across all options)
11: added four different hospital triaging policies, keeping the total care provided constant across the options
    (that is, the total care provided only depends on the level of overcapacity of the hospital)
12: changed model to (by default) always calculate R0, runs for a random choice in the parameter space
13: generalized formulas so that cases where p_IH_young!=0 and p_HD_young!=0 can be considered,
    added a preliminary version of testing in the model (introduced tested as another attribute just like ages), 
    anybody tested positive engages in a strong reduction of all activity but no testing policies yet
14: skipped
15: testing_policy_decision_minor works, testing_policy_decision_major (old vs young), not yet
16: (still missing: testing_policy_decision_major (old vs young) and reduction_old), works
17: runs a huge number of single simulations, completely uniformly sampling from the parameter space
18: corrected a mistake in the calculation of R0, before in some instances R0==0,deaths>1 could be reported
    because the calculation of R0 was stopped too early (i.e. when patient 0 was still infectious),
    corrected the way contact probabilities are calculated, before the geometric mean was used to combine the contact
    rate between two people, now we use the arithmetric mean
19: corrected another mistake in the calculation of R0, now it is correct: If R0=0, nobody gets infected, otherwise R0>=1,
    changed the way edge weights are calculated, now using the law of mass action and not sqrt(law of mass action)
20: added recorded of peak number of infected, hospitalized and symptomatic witthout testing
21: reduced the baseline contact rate: the private small-world interaction network still has k average connections per node 
    but each connection (edge) is only active on a given day with a certain probability,
    accordingly adjusted the probability of public interactions to keep same average numbers of private and public interactions
22: deleted the explicit modeling of the rate of contact introduced in v21, instead b_I and b_A implicitly correspond to 
    the transmission rate, which is the rate of contact times the probability of transmission (given contact).
    Changed the transition times from Poisson to continuous distributions found in the literature using a int(round(x)) to discretize
23: k varies modeling low contact countries (k=4, Germany), mediocre (k=7), and high (k=10, Italy)
24: skipped
25: skipped
26: final version of model, varies k (k needs to be even for small-world networks!)'''

version='26'

#built-in modules
import sys
import random
import math

#added modules
import numpy as np
import networkx as nx
#import itertools
from scipy.interpolate import interp1d

output_folder = 'results/model%s/' % version

try:
    filename = sys.argv[0]
    SLURM_ID = int(sys.argv[1])
except:
    filename = ''
    SLURM_ID = -1

if len(sys.argv)>2:
    nsim = int(sys.argv[2])
else:
    nsim = 10
    
## parameters for network generation
if len(sys.argv)>3:
    N = int(sys.argv[3]) #network size
else:
    N = 1000
   
if len(sys.argv)>4:
    k = int(sys.argv[4]) #average number of edges per node, initially nodes are connected in a circle, k/2 neighbors to the left, k/2 to the right
else:
    k = 6

if len(sys.argv)>5:
    p_edgechange = float(sys.argv[5]) #average number of edges per node, initially nodes are connected in a circle, k/2 neighbors to the left, k/2 to the right
else:
    p_edgechange = 0.05
    
if len(sys.argv)>6:
    p_old = float(sys.argv[6]) #proportion of old people, p_young = 1-p_old
else:
    p_old = 1/3
    
#network_generating_function = nx.newman_watts_strogatz_graph

## parameters for overall simulation
OUTPUT = 0#True
T_max = 1000 #in days

def get_optimal_per_day_death_rate(params_HR,p_HD=0.01,GET_INTERPOLATOR=False):
    #X ~ Poisson(t_HR), Y ~ Geom(d), fit the per day death rate d such that P(X>Y) = p_HD
    t_HR = params_HR if type(params_HR) in [float,int] else params_HR[0]
    fak = 1
    prob_X_is_k = []
    upto = 8*t_HR
    for k in range(upto):
        prob_X_is_k.append( t_HR**k*math.exp(-t_HR)/fak)
        fak*=(k+1)
    ps = np.arange(0.0001,0.2,0.001)
    p_Y_less_than_Xs = []
    for per_day_death_rate_hospital in ps:
        prob_Y_less_than_k = [0]
        for k in range(1,upto):
            prob_Y_less_than_k.append(1 - (1-per_day_death_rate_hospital)**k)
        p_Y_less_than_X = np.dot(prob_X_is_k,prob_Y_less_than_k)
        p_Y_less_than_Xs.append(p_Y_less_than_X)
    p_Y_less_than_Xs = np.array(p_Y_less_than_Xs)
    #np.interp(p_Y_less_than_Xs,ps)
    f2 = interp1d(p_Y_less_than_Xs, ps, kind='cubic')
    if GET_INTERPOLATOR:
        return f2
    else:
        return f2([p_HD])[0]

##Functions modeling random transitions
def get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI):
    if p_A>random.random():
        nextstage = 'A'
        timeto = max(1,int(round(np.random.poisson(*params_EA))))
    else:
        nextstage = 'I'
        timeto = max(1,int(round(np.random.poisson(*params_EI))))
    return (nextstage,timeto)

def get_random_course_of_infection_I_to_H_or_R(p_H,params_IH,params_IR):
    if p_H>random.random():
        nextstage = 'H'
        timeto = max(1,int(round(np.random.poisson(*params_IH))))
    else:
        nextstage = 'R'
        timeto = max(1,np.random.poisson(params_IR))
    return (nextstage,timeto)

def get_random_time_until_recovery_if_asymptomatic(params_AR):
    timeto = np.random.poisson(params_AR)
    nextstage = 'RA' 
    return (nextstage,timeto)

def get_random_time_until_recovery_under_perfect_care(params_HR):
    timeto = np.random.poisson(params_HR)
    nextstage = 'Unknown' #might die each day in hospital due to hospital-associated per-day death rate
    return (nextstage,timeto)

shape=2
scale=2
transmission_rate_over_time = np.array([t**(shape-1)*np.exp(-t/scale) for t in range(1,50+1)])
normalized_transmission_rate_over_time = transmission_rate_over_time/max(transmission_rate_over_time)

def get_transmission_probs(t_EAI,b_AI,activity_reduction_due_to_age):#could make this expoentially increasing but that requires knowledge of how the viral load increases
    return [0 for t in range(1,int(t_EAI)-1)] + list(activity_reduction_due_to_age*b_AI*normalized_transmission_rate_over_time)

def total_edge_weight_in_susceptible_network(network,ages,activity_old,private_activity_SEA,public_activity_SEA,c,n_interactions_old_old_private=None,n_interactions_old_young_private=None,n_interactions_young_young_private=None):
    #c = probability_of_close_contact_with_random_person_in_public
    N = len(ages)
    
    if n_interactions_old_old_private==None or n_interactions_young_young_private==None or n_interactions_young_young_private==None:
        #private interactions
        n_interactions_old_old_private,n_interactions_old_young_private,n_interactions_young_young_private = 0,0,0
        for edge in network.edges():
            if ages[edge[0]] == ages[edge[1]] and ages[edge[0]]==1:
                n_interactions_old_old_private += 1
            elif ages[edge[0]] == ages[edge[1]] and ages[edge[0]]==0:
                n_interactions_young_young_private += 1
            else:
                n_interactions_old_young_private += 1
    edge_weight_between_two_young_private = private_activity_SEA**2
    total_edge_weight_private = edge_weight_between_two_young_private*(n_interactions_young_young_private + n_interactions_old_young_private*activity_old + n_interactions_old_old_private*activity_old*activity_old)        
    max_total_edge_weight_private = n_interactions_old_old_private + n_interactions_old_young_private + n_interactions_young_young_private
        
    #public interactions
    edge_weight_between_two_young_public = public_activity_SEA**2
    n_old = sum(ages)
    n_interactions_old_young_public = n_old*(N-n_old)
    n_interactions_old_old_public = n_old*(n_old-1)/2
    n_interactions_young_young_public = (N-n_old)*(N-n_old-1)/2
    total_edge_weight_public = c*edge_weight_between_two_young_public*(n_interactions_young_young_public + n_interactions_old_young_public*activity_old + n_interactions_old_old_public*activity_old*activity_old)
    max_total_edge_weight_public = c*N*(N-1)/2
    
    return (total_edge_weight_private+total_edge_weight_public)/(max_total_edge_weight_private+max_total_edge_weight_public),total_edge_weight_private,total_edge_weight_public,n_interactions_old_old_private,n_interactions_old_young_private,n_interactions_young_young_private

def model(N,k,p_edgechange,network_generating_function,p_old,T_max,b_A,b_I,b_H,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_young_over_p_A_old,p_IH,p_IH_young_over_p_IH_old,overall_death_rate_covid19,p_HD_young_over_p_HD_old,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,private_activity_I,private_activity_H,private_old_young_activity,public_activity_SEA,activity_old,public_activity_I,public_activity_H,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,seed=None,OUTPUT=False,ESTIMATE_R0=True,interpolator_per_day_death_rate=None):
    if seed==None:
        seed = np.random.randint(1,2**32 - 1)
    #seed = np.random.randint(1,2**32 - 1) #delete this!!!!!!
    random.seed(seed)
    np.random.seed(seed)
    
    
    #built local interaction network and create a list of lists, called neighbors
    network = network_generating_function(N,k,p_edgechange)
    #to draw network: nx.draw(network)
    neighbors = [[] for _ in range(N)]
    for (a,b) in network.edges():
        neighbors[a].append(b)
        neighbors[b].append(a)

    #randomly distribute the number of old people, old=1, young=0
    ages = np.random.random(N)<p_old
    activities_old = [activity_old if ages[i]==1 else 1 for i in range(N)]

    total_edge_weight = total_edge_weight_in_susceptible_network(network,ages,activity_old,private_activity_SEA,public_activity_SEA,probability_of_close_contact_with_random_person_in_public)[0]

    params_EA = [params_EA] if type(params_EA) in [float,int] else params_EA #mean of Poisson RV
    params_EI = [params_EI] if type(params_EI) in [float,int] else params_EI #mean of Poisson RV
    params_IH = [params_IH] if type(params_IH) in [float,int] else params_IH #mean of Poisson RV
    params_AR = [params_AR] if type(params_AR) in [float,int] else params_AR #mean of Poisson RV
    params_IR = [params_IR] if type(params_IR) in [float,int] else params_IR #mean of Poisson RV
    params_HR = [params_HR] if type(params_HR) in [float,int] else params_HR #mean of Poisson RV

    nextstage = ['E' for _ in range(N)]
    currentstage = ['S' for _ in range(N)]
    time_to_nextstage = np.array([-1 for _ in range(N)],dtype=float)

    p_A_old = p_A/(p_A_young_over_p_A_old*(1-p_old)+p_old) #calculate the probability that an old exposed person will have an asymptomatic infection
    p_A_young = p_A_young_over_p_A_old*p_A_old #calculate the probability that a young exposed person will have an asymptomatic infection
    p_IH_old = (1-p_A)*p_IH/(  p_old*(1-p_A_old) + p_IH_young_over_p_IH_old*(1-p_old)*(1-p_A_young)  )
    p_IH_young = p_IH_young_over_p_IH_old*p_IH_old 
    p_HD = overall_death_rate_covid19/(p_old*(1-p_A_old)*p_IH_old + (1-p_old)*(1-p_A_young)*p_IH_young)
    p_HD_old = (1-p_A)*p_IH*p_HD/(  p_old*(1-p_A_old)*p_IH_old + p_HD_young_over_p_HD_old*(1-p_old)*(1-p_A_young)*p_IH_young  )
    p_HD_young = p_HD_young_over_p_HD_old*p_HD_old 

    if interpolator_per_day_death_rate!=None and interpolator_per_day_death_rate[0]==params_HR:
        pass
    else:
        interpolator_per_day_death_rate = [params_HR,get_optimal_per_day_death_rate(params_HR,0.01,True)]
    per_day_death_rate_hospital_old = interpolator_per_day_death_rate[1]([p_HD_old])[0] #Bernoulli (death in hospital is geometrically distributed, the longer one stays in the hospital the more likely to die)
    per_day_death_rate_hospital_young = interpolator_per_day_death_rate[1]([p_HD_young])[0] #Bernoulli (death in hospital is geometrically distributed, the longer one stays in the hospital the more likely to die)

    #S-E-I-R or S-E-I-H-R or S-E-I-H-D
    #S=susceptible, E=exposed, A=asymptomatic, I=infected, H=hospitalized, R=recovered, D=dead
    initial_exposed = random.randint(0,N-1)
    S = list(range(N))
    S.pop(initial_exposed)
    p_A = p_A_old if ages[initial_exposed]==1 else p_A_young
    nextstage[initial_exposed],time_to_nextstage[initial_exposed] = get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI)
    currentstage[initial_exposed] = 'E'
    dict_transmission_probs = dict({initial_exposed:get_transmission_probs(time_to_nextstage[initial_exposed],b_I if nextstage[initial_exposed]=='I' else b_A,activity_old if ages[initial_exposed]==1 else 1)})
    E = [initial_exposed]
    A = []
    I = []
    R = [] #R=RP, recovered not in RA were symptomatic and assume that they had COVID19
    RA = []
    RP = []
    H = [] 
    D = []        

    len_E_old = 1 if ages[initial_exposed]==1 else 0
    len_A_old = 0
    len_I_old = 0
    len_H_old = 0
    len_E_young = 1 if ages[initial_exposed]==0 else 0
    len_A_young = 0
    len_I_young = 0
    len_H_young = 0
    
    time_infected = [np.nan for i in range(N)]
    time_infected[initial_exposed] = 0
    disease_generation_time = [np.nan for i in range(N)]
    infections_caused_byE,infections_caused_byA,infections_caused_byI = 0,0,0


    #initially nobody is tested
    #tested_positive = np.array(np.round(np.random.random(N)),dtype=bool)
    tested_positive = np.zeros(N,dtype=bool)
    #initially nobody will be tested
    TEST=False
    time_to_test_result = [-1 for i in range(N)]

    I_not_tested_old = []
    I_not_tested_young = []
    H_not_tested_old = []
    H_not_tested_young = []
    len_I_not_tested_old = len(I_not_tested_old)
    len_I_not_tested_young = len(I_not_tested_young)
    len_H_not_tested_old = len(H_not_tested_old)
    len_H_not_tested_young = len(H_not_tested_young)
    I_waiting = []
    H_waiting = []
    len_IP_old=0
    len_HP_old=0    
    len_AP_old=0
    len_EP_old=0
    len_IP_young=0
    len_HP_young=0    
    len_AP_young=0
    len_EP_young=0

    max_len_H,max_len_I,max_len_not_tested = 0,0,0
    
    if triaging_policy in ['FBLS','FBrnd']:
        currently_in_care = []

    if OUTPUT:
        res = [[len(S),len(E),len(A),len(I),len(H),len(R),len(RA),len(RP),len(D)]]
        
    if ESTIMATE_R0:
        counter_secondary_infections = [0 for i in range(N)]
    else:
        counter_secondary_infections = [np.nan for i in range(N)]
    
    for t in range(1,T_max+1):
        #things that happen on any day (synchronously):
        #1. All S can get infected via private or public interactions,
        #   this happens with a certain probability based on activity levels (social distancing policies)
        #2. All newly infected S move to E,
        #   and it is determined (Bernoulli random variable) if they continue to move to A or I as well as the next transition time (Poisson)
        #3. All E, I, A move one day "closer" to the next compartment (E->I/A, I->H/R, A->RA), 
        #   if they "reach" the next compartment, a Poisson-distributed transition time to the next compartment and,
        #   possibly, a random Bernoulli variable deciding which compartment is drawn
        #4. All H have a risk of dying (Bernoulli random variable), 
        #5. All H move closer to R (the "distance" they move closer depends on the triaging_policy and hospital overcapacity),
        #   If they "reach" R, they move to R and are recovered.
        #   Note: The risk of dying is proportionally reduced for individuals that don't require a full day for recovery.
        #6. Any person may get tested (testing policies decide who gets limited tests).
        #7. Any person may receive a test result (possibly delayed),
        #   and if positive, move to a special category (E->EP, A->AP, I->IP, H->HP, R->RP). 
        #   Positive tests significantly reduce activity levels, both public and private, due to quarantine, however not to 0 due to imperfect quarantine measures taken by the average person
        #   Note: There is no category SP because FPR is assumed to be 0. 
        #   Further, there are no negative test categories because a negative test "today" does not exclude a positive test "tomorrow".

        #print(t,len_E,len(E),len_I,len(I),len_A,len(A),len_H,len(H))# - delete at end
        
        #start testing once the first sympytomatic person presents at the hospital
        if len_I_old+len_I_young>0 and TEST==False:
            TEST=True
        
        if TEST:
            assert testing_policy_decision_minor in ['random','FIFT','LIFT']
            assert testing_policy_decision_major in ['O>Y','Y>O']
            #pick who to test based on policies, need to know for which category who has not been tested,
            #test results get back at the earliest at the end of day (if delay = 0) so this day these ppl can still infect others
            tests_left = max_number_of_tests_available_per_day
            order_H,order_I = 0,1
            order_old,order_young = (0,1) if testing_policy_decision_major=='O>Y' else (1,0)
            counter_major = 0
            counter_minor = 0
            while tests_left>0 and (len_H_not_tested_old>0 or len_I_not_tested_old>0 or len_H_not_tested_young>0 or len_I_not_tested_young>0):
                if len_H_not_tested_old>0 and counter_minor==order_H and counter_major==order_old:
                    if testing_policy_decision_minor =='random':
                        h = H_not_tested_old.pop(int(random.random()*len_H_not_tested_old))
                    elif testing_policy_decision_minor =='FIFT':
                        h = H_not_tested_old.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        h = H_not_tested_old.pop()
                    time_to_test_result[h] = testing_delay+1
                    len_H_not_tested_old -= 1
                    H_waiting.append(h)
                    tests_left -= 1
                elif len_H_not_tested_old==0 and counter_minor==order_H and counter_major==order_old: #ran out of H to be tested
                    counter_minor+=1
                elif len_H_not_tested_young>0 and counter_minor==order_H and counter_major==order_young:
                    if testing_policy_decision_minor =='random':
                        h = H_not_tested_young.pop(int(random.random()*len_H_not_tested_young))
                    elif testing_policy_decision_minor =='FIFT':
                        h = H_not_tested_young.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        h = H_not_tested_young.pop()
                    time_to_test_result[h] = testing_delay+1
                    len_H_not_tested_young -= 1
                    H_waiting.append(h)
                    tests_left -= 1
                elif len_H_not_tested_young==0 and counter_minor==order_H and counter_major==order_young: #ran out of H to be tested
                    counter_minor+=1             
                elif len_I_not_tested_old>0 and counter_minor==order_I and counter_major==order_old:
                    if testing_policy_decision_minor =='random':
                        i = I_not_tested_old.pop(int(random.random()*len_I_not_tested_old))
                    elif testing_policy_decision_minor =='FIFT':
                        i = I_not_tested_old.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        i = I_not_tested_old.pop()
                    time_to_test_result[i] = testing_delay+1
                    len_I_not_tested_old -= 1
                    I_waiting.append(i)
                    tests_left -= 1
                elif len_I_not_tested_old==0 and counter_minor==order_I and counter_major==order_old: #ran out of I to be tested
                    counter_minor+=1
                elif len_I_not_tested_young>0 and counter_minor==order_I and counter_major==order_young:
                    if testing_policy_decision_minor =='random':
                        i = I_not_tested_young.pop(int(random.random()*len_I_not_tested_young))
                    elif testing_policy_decision_minor =='FIFT':
                        i = I_not_tested_young.pop(0)
                    elif testing_policy_decision_minor =='LIFT':
                        i = I_not_tested_young.pop()
                    time_to_test_result[i] = testing_delay+1
                    len_I_not_tested_young -= 1
                    I_waiting.append(i)
                    tests_left -= 1
                elif len_I_not_tested_young==0 and counter_minor==order_I and counter_major==order_young: #ran out of I to be tested
                    counter_minor+=1  
                if counter_minor==2:
                    counter_minor = 0
                    counter_major += 1
                    

        if TEST:
            #see who gets test results back
            for i in I_waiting[:]:
                time_to_test_result[i] -= 1
                if time_to_test_result[i]==0: #test results come back
                    dict_transmission_probs[i] = [activity_P*el for el in dict_transmission_probs[i]]
                    I_waiting.remove(i)
                    tested_positive[i]=True
                    if ages[i]==1:
                        len_IP_old+=1
                    else:
                        len_IP_young+=1                        
            for h in H_waiting[:]:
                time_to_test_result[h] -= 1
                if time_to_test_result[h]==0: #test results come back
                    dict_transmission_probs[h] = [activity_P*el for el in dict_transmission_probs[h]]
                    H_waiting.remove(h)
                    tested_positive[h]=True
                    if ages[h]==1:
                        len_HP_old+=1
                    else:
                        len_HP_young+=1                     
        
        
        #go through all exposed, asymptomatic, symptomatic and hospitalized and check whether they cause new infections among their private or random public contacts
        dict_newly_exposed = {}

        prob_public_infection_old_S = []
        prob_public_infection_young_S = []
        for ii,(contagious_compartment,private_activity_level_contagious,public_activity_level_contagious) in enumerate(zip([E,A,I,H],[private_activity_SEA,private_activity_SEA,private_activity_I,private_activity_H],[public_activity_SEA,public_activity_SEA,public_activity_I,public_activity_H])):
            for contagious in contagious_compartment:
                try:
                    current_transmission_prob = dict_transmission_probs[contagious].pop(0)
                except IndexError: #same patients can stay really long in a potentially contagious category, especially H, set infectivity to 0 then.
                    current_transmission_prob = 0
                private_activity_times_infectiousness = private_activity_level_contagious*current_transmission_prob
                if private_activity_level_contagious>0:
                    for neighbor in neighbors[contagious]:
                        #private/local contacts:
                        if nextstage[neighbor]=='E':
                            probability_of_infection = private_activity_times_infectiousness * (private_activity_SEA*activities_old[neighbor]) * (private_old_young_activity if ages[neighbor]!=ages[contagious] else 1)
                            if probability_of_infection>random.random():                           
                                try:
                                    dict_newly_exposed[neighbor].append(contagious)
                                except KeyError:
                                    dict_newly_exposed.update({neighbor:[contagious]})
                #public/random contacts:
                public_activity_times_infectiousness = public_activity_level_contagious*current_transmission_prob
                prob_public_infection_young_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA))
                prob_public_infection_old_S.append(probability_of_close_contact_with_random_person_in_public*public_activity_times_infectiousness*(public_activity_SEA*activity_old))
        probability_that_young_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_young_S])
        probability_that_old_S_gets_infected_publicy = 1-np.prod([1-el for el in prob_public_infection_old_S])
        list_all_contagious = E+A+I+H
                
        for s in S:
            if ages[s]==1 and probability_that_old_S_gets_infected_publicy>random.random():
                who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_old_S)[0]
                try:
                    dict_newly_exposed[s].append(who_did_it)
                except KeyError:
                    dict_newly_exposed.update({s:[who_did_it]})
            elif ages[s]==0 and probability_that_young_S_gets_infected_publicy>random.random():
                who_did_it = random.choices(list_all_contagious,weights=prob_public_infection_young_S)[0]
                try:
                    dict_newly_exposed[s].append(who_did_it)
                except KeyError:
                    dict_newly_exposed.update({s:[who_did_it]})                                                 
                        
        #print(len(E),len(A),len(I),drivenby,dict_newly_exposed)

        #hospitalized get perfect care (progress 1 day closer to recovery) if hospitals aren't overrun, if overrun this decreases
        len_H = len_H_old+len_H_young
        capacity = len_H/N/hospital_beds_per_person# 1= at capacity, 0=compeltely empty, 2=200%, 3=300%
        avg_care_available_per_person = min(1,capacity**(-care_decline_exponent)) if capacity>0 else 1 #could also use capacity**(-.8) or capacity**(-.5), then total care provided wouldnt be constant anymore though
        total_care_provided = avg_care_available_per_person*len_H
        
        assert triaging_policy in ['FCFS','same','LSF','FBLS','FBrnd']
        if triaging_policy=='same':
            total_care_left = total_care_provided
            nr_without_care_thus_far = len_H
            without_care_thus_far = [True for _ in range(len_H)]
            care_provided_for_each_person = [0 for _ in range(len_H)]
            while total_care_left>0 and nr_without_care_thus_far>0:
                avg_additional_care_available_per_person = total_care_left/nr_without_care_thus_far
                for ii,h in enumerate(H):
                    if without_care_thus_far[ii]:
                        if care_provided_for_each_person[ii]+avg_additional_care_available_per_person>=time_to_nextstage[h]:
                            care_provided_for_each_person[ii] = time_to_nextstage[h]
                            nr_without_care_thus_far-=1
                            without_care_thus_far[ii]=False
                        else:
                            care_provided_for_each_person[ii] += avg_additional_care_available_per_person
                total_care_distributed = sum(care_provided_for_each_person)
                total_care_left = total_care_provided-total_care_distributed
        else: #i.e., if 'FCFS','LSF','FBLS'
            if triaging_policy=='LSF':
                time_to_recovery_under_optimal_care = [time_to_nextstage[h] for h in H]
                #resort H
                H = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care,H))]
            elif triaging_policy=='FBLS': #currently_in_care is a stack
                time_to_recovery_under_optimal_care = [time_to_nextstage[h] for h in H]
                nr_currently_in_care = len(currently_in_care)
                total_care_required_by_those_in_care = sum([min(1,time_to_nextstage[h]) for h in currently_in_care])
                if total_care_required_by_those_in_care>total_care_provided:
                    #resort those currently in care based on severity of infection
                    currently_in_care = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care[:nr_currently_in_care],H[:nr_currently_in_care]))]
                    H = currently_in_care + H[nr_currently_in_care:]            
                else:
                    #resort H so that those who are currently in care are at the front and the remaining ones are sorted based on severity of infection
                    remainder_of_H_sorted = [h for _,h in sorted(zip(time_to_recovery_under_optimal_care,H[nr_currently_in_care:]))]
                    H = currently_in_care + remainder_of_H_sorted
            elif triaging_policy=='FBrnd': #currently_in_care is a stack
                nr_currently_in_care = len(currently_in_care)
                not_in_care = H[nr_currently_in_care:] 
                random.shuffle(not_in_care) #shuffles in place
                H = currently_in_care + not_in_care
            care_provided_for_each_person = []
            total_care_left = total_care_provided
            for h in H:
                if total_care_left>=1:
                    care_for_this_person = min(1,time_to_nextstage[h])
                elif total_care_left==0:
                    care_for_this_person = 0
                else:
                    care_for_this_person = min(total_care_left,time_to_nextstage[h])
                total_care_left -= care_for_this_person
                care_provided_for_each_person.append(care_for_this_person)                
        
        #if sum(drivenby)>0:
        #    new_infections_caused_by = 1/sum(drivenby)*len(dict_newly_exposed)
        #    print(t,len(E),len(A),len(I),np.array(drivenby)*new_infections_caused_by,len(dict_newly_exposed))

        #disease generation time and who caused infections
        for newly_exposed in dict_newly_exposed: 
            len_dict_newly_exposed = len(dict_newly_exposed[newly_exposed])
#            if ESTIMATE_R0 or OUTPUT:
            for contagious in dict_newly_exposed[newly_exposed]:
                counter_secondary_infections[contagious] += 1/len_dict_newly_exposed
                if currentstage[contagious]=='E':
                    infections_caused_byE += 1/len_dict_newly_exposed
                elif currentstage[contagious]=='A':
                    infections_caused_byA += 1/len_dict_newly_exposed
                elif currentstage[contagious]=='I':
                    infections_caused_byI += 1/len_dict_newly_exposed
                
            if len_dict_newly_exposed>1:
                dummy = []
                for el in dict_newly_exposed[newly_exposed]:
                    dummy.append(t-time_infected[el])
                disease_generation_time[newly_exposed] = np.mean(dummy)
            else:
                disease_generation_time[newly_exposed] = t-time_infected[dict_newly_exposed[newly_exposed][0]]
                time_infected[newly_exposed] = t
                       
        
        #see if hospitalized die (do this simultaneously while transitioning closer to recovery based on care received
        #and checking for recovery), 
        #also: reduce death rate for those that recover that day
        for ii,h in enumerate(H[:]):
            time_left_until_recovery = time_to_nextstage[h]-care_provided_for_each_person[ii]
            part_of_day_that_care_is_required_until_recovery = 1 if time_left_until_recovery>0 else care_provided_for_each_person[ii]
            per_day_death_rate_hospital = per_day_death_rate_hospital_old if ages[h]==1 else per_day_death_rate_hospital_young
            if per_day_death_rate_hospital*part_of_day_that_care_is_required_until_recovery>random.random():#hospitalized die at certain rate
                nextstage[h] = 'D'
                currentstage[h] = 'D'
                H.remove(h)
                D.append(h)
                time_to_nextstage[h]=t #keeps track of when final stage was reached
                if tested_positive[h]:
                    if ages[h]==1:
                        len_HP_old-=1
                    else:
                        len_HP_young-=1
                else:
                    try: #if dying person was waiting for test result
                        H_waiting.remove(h)
                        tested_positive[h]=True
                    except ValueError:
                        if ages[h]==1:#old
                            len_H_not_tested_old -= 1
                            H_not_tested_old.remove(h)
                        else:
                            len_H_not_tested_young -= 1
                            H_not_tested_young.remove(h)
                if ages[h]==1:
                    len_H_old-=1
                else:
                    len_H_young-=1 
                if triaging_policy in ['FBLS','FBrnd']:
                    try:
                        currently_in_care.remove(h)
                    except ValueError:#patient that died never received hospital care due to overcapacity
                        pass            
              #hospitalized get better at certain rate, dependent of care level  
            elif time_left_until_recovery<=0:
                H.remove(h)
                currentstage[h]='R'
                nextstage[h] = 'R'
                if ages[h]==1:
                    len_H_old-=1
                else:
                    len_H_young-=1                    
                time_to_nextstage[h] = t #set this to t to have a counter that keeps track of when recovery happened
                if tested_positive[h]:
                    if ages[h]==1:
                        len_HP_old-=1
                    else:
                        len_HP_young-=1                        
                    RP.append(h)
                    
                else:
                    try: #if recovered person was waiting for test result
                        H_waiting.remove(h)
                        RP.append(h)
                        tested_positive[h]=True
                    except ValueError:
                        R.append(h)
                        if ages[h]==1:#old
                            len_H_not_tested_old -= 1
                            H_not_tested_old.remove(h)
                        else:
                            len_H_not_tested_young -= 1
                            H_not_tested_young.remove(h)
            else:
                time_to_nextstage[h] = max(0,time_left_until_recovery)
                    
                    
        #transition one day closer towards next stage
        for index in E+A+I:
            time_to_nextstage[index] = max(0,time_to_nextstage[index]-1)
        
        #Careful!!!! Need to execute this going from last stage to first stage to avoid logical errors
        for i in I[:]:
            if time_to_nextstage[i]==0:
                if ages[i]==1:
                    len_I_old-=1
                else:
                    len_I_young-=1                    
                if nextstage[i]=='H':
                    currentstage[i]='H'
                    I.remove(i)
                    H.append(i)
                    nextstage[i],time_to_nextstage[i] = get_random_time_until_recovery_under_perfect_care(params_HR)
                    if tested_positive[i]:
                        if ages[i]==1:
                            len_IP_old-=1
                            len_HP_old+=1
                        else:
                            len_IP_young-=1
                            len_HP_young+=1                            
                    else:
                        try: #if hospitalized person was already waiting for test result
                            I_waiting.remove(i)
                            H_waiting.append(i)
                        except ValueError:
                            if ages[i]==1:#old
                                len_I_not_tested_old -= 1
                                len_H_not_tested_old += 1
                                H_not_tested_old.append(i)
                                I_not_tested_old.remove(i) 
                            else:
                                len_I_not_tested_young -= 1
                                len_H_not_tested_young += 1
                                H_not_tested_young.append(i)
                                I_not_tested_young.remove(i)                                 
                    if ages[i]==1:
                        len_H_old+=1
                    else:
                        len_H_young+=1                        
                elif nextstage[i]=='R':  
                    currentstage[i]='R'
                    I.remove(i)
                    if tested_positive[i]:
                        if ages[i]==1:
                            len_IP_old-=1
                        else:
                            len_IP_young-=1                            
                        RP.append(i)
                    else:
                        try: #if hospitalized person was already waiting for test result
                            I_waiting.remove(i)
                            RP.append(i)
                            tested_positive[i]=True
                        except ValueError:
                            R.append(i)
                            if ages[i]==1:#old
                                len_I_not_tested_old -= 1
                                I_not_tested_old.remove(i)
                            else:
                                len_I_not_tested_young -= 1
                                I_not_tested_young.remove(i)                                
                    time_to_nextstage[i] = t #set this to t to have a counter that keeps track of when recovery happened
                  
        for a in A[:]:
            if time_to_nextstage[a]==0:  
                
                A.remove(a)
                if tested_positive[a]:#if tested positive, this person knew they had COVID19 so they go into R, otherwise RA (might get tested randomly)
                    if ages[a]==1:
                        len_AP_old-=1
                    else:
                        len_AP_young-=1
                    RP.append(a)
                    currentstage[a]='RP'
                else:
                    RA.append(a)
                    currentstage[a]='RA'
                if ages[a]==1:
                    len_A_old-=1
                else:
                    len_A_young-=1
                time_to_nextstage[a] = t #set this to t to have a counter that keeps track of when recovery happened

        for e in E[:]: #check for changes, i.e. for time_to_nextstage[e]==0  
            if time_to_nextstage[e]==0:
                #del dict_transmission_probs[e]#dict_transmission_probs.pop(e)
                if nextstage[e]=='I': #current stage must be E
                    currentstage[e]='I'
                    E.remove(e)
                    I.append(e) 
                    if tested_positive[e]:
                        if ages[e]==1:
                            len_EP_old-=1
                            len_IP_old+=1
                        else:
                            len_EP_young-=1
                            len_IP_young+=1                            
                    else:
                        if ages[e]==1:#old
                            I_not_tested_old.append(e)
                            len_I_not_tested_old+=1
                        else:
                            I_not_tested_young.append(e)
                            len_I_not_tested_young+=1 
                    if ages[e]==1:
                        len_E_old-=1
                        len_I_old+=1
                    else:
                        len_E_young-=1
                        len_I_young+=1                        
                    p_IH = p_IH_old if ages[e]==1 else p_IH_young
                    nextstage[e],time_to_nextstage[e] = get_random_course_of_infection_I_to_H_or_R(p_IH,params_IH,params_IR)
                elif nextstage[e]=='A': #current stage must be E
                    currentstage[e]='A'
                    E.remove(e)
                    A.append(e)
                    if tested_positive[e]:
                        if ages[e]==1:
                            len_EP_old-=1
                            len_AP_old+=1
                        else:
                            len_EP_young-=1
                            len_AP_young+=1 
                    if ages[e]==1:
                        len_E_old-=1
                        len_A_old+=1
                    else:
                        len_E_young-=1
                        len_A_young+=1                        
                    nextstage[e],time_to_nextstage[e] = get_random_time_until_recovery_if_asymptomatic(params_AR)
                    
        #go through all newly exposed and move them from S to E, and randomly pick their next stage
        for newly_exposed in dict_newly_exposed: 
            S.remove(newly_exposed)
            E.append(newly_exposed)
            currentstage[newly_exposed]='E'
            if ages[newly_exposed]==1:
                len_E_old+=1
            else:
                len_E_young+=1                
            p_A = p_A_old if ages[newly_exposed]==1 else p_A_young
            nextstage[newly_exposed],time_to_nextstage[newly_exposed] = get_random_course_of_infection_E_to_A_or_I(p_A,params_EA,params_EI)
            dict_transmission_probs.update({newly_exposed:get_transmission_probs(time_to_nextstage[newly_exposed],b_I if nextstage[newly_exposed]=='I' else b_A,activity_old if ages[newly_exposed] else 1)})

        if len_H_old+len_H_young>max_len_H:
            max_len_H = len_H_old+len_H_young
            
        if len_I_old+len_I_young>max_len_I:
            max_len_I = len_I_old+len_I_young
            
        len_not_tested = len_I_not_tested_old+len_I_not_tested_young+len_H_not_tested_old+len_H_not_tested_young
        if len_not_tested>max_len_not_tested:
            max_len_not_tested = len_not_tested

        if OUTPUT:
            res.append([len(S),len(E),len(A),len(I),len(H),len(R),len(RA),len(RP),len(D)])
            
#        if ESTIMATE_R0 and not OUTPUT:
#            if (initial_exposed in R or initial_exposed in RA or initial_exposed in RP or initial_exposed in D):
#                #initial case is done infecting more people
#                ESTIMATE_R0 = False        
           
        if I==[] and E==[] and H==[] and A==[]:
            if OUTPUT:
                print('stopped after %i iterations because there were no more exposed or infected or hospitalized' % t)
            #assert not ESTIMATE_R0,'should never get here because either the initial_exposed will end up in R or D'
            if OUTPUT:
                return res,seed,counter_secondary_infections,time_infected,disease_generation_time,total_edge_weight,max_len_H,max_len_I,max_len_not_tested
            else:
                return len(S),len(R),len(RA),len(RP),len(D),seed,counter_secondary_infections[initial_exposed],np.nanmean(disease_generation_time),np.nanmean(time_infected),infections_caused_byE,infections_caused_byA,infections_caused_byI,total_edge_weight,max_len_H,max_len_I,max_len_not_tested
    #ssert not ESTIMATE_R0,'should never get here because the initial_exposed should long be in R or D'
    if OUTPUT:
        return res,seed,counter_secondary_infections,time_infected,disease_generation_time,total_edge_weight,max_len_H,max_len_I,max_len_not_tested
    else:
        return len(S),len(R),len(RA),len(RP),len(D),seed,counter_secondary_infections[initial_exposed],np.nanmean(disease_generation_time),np.nanmean(time_infected),infections_caused_byE,infections_caused_byA,infections_caused_byI,total_edge_weight,max_len_H,max_len_I,max_len_not_tested


#fixed parameters
#N,k,p_edgechange,p_old (and nsim) as command line arguments

network_generating_function = nx.watts_strogatz_graph #can theoretically also use nx.newman_watts_strogatz_graph but then there is not the average same number of private and public interactions
params_EA = 5 #mean of Poisson RV
params_EI = 5 #mean of Poisson RV
params_IH = 8 #mean of Poisson RV
params_AR = 20 #mean of Poisson RV
params_IR = 20 #mean of Poisson RV
params_HR = 12 #mean of Poisson RV
p_IH = 0.07 #P(H|I) = 1 - P(R|I), probability of eventually moving from symptomatic class (I) to hospitalized class (H)
overall_death_rate_covid19 = 0.01
activity_reduction_H = 1
hospital_beds_per_person = 3/1000 * 2
care_decline_exponent = 0.5
private_old_young_activity = 1#1=no change in behavior, 0=all contacts stopped; don't analyze distancing between old/young, rather focus on distancing of old 

#randomly sampled from parameter space
initial_seed = np.random.randint(1,2**32 - 1)

#Note: There are further parameters that are derived per model run (see beginning of model)

#to speed things up, interpolate the per day death rate once and fit every time using this interpolator, works as long as the parameters params_HR are constant, which will be checked
interpolator_per_day_death_rate = [params_HR,get_optimal_per_day_death_rate(params_HR,0.01,True)]

neverinfected_counts = []
death_counts = []
R0s = []
mean_generation_times = []
mean_time_infections = []
infections_caused_byEs = []
infections_caused_byAs = []
infections_caused_byIs = []
total_initial_edge_weights = []
max_len_Is = []
max_len_Hs = []
max_len_not_testeds = []


args = []
for i in np.arange(nsim):
    #randomly sampled from parameter space
    b_I = np.random.uniform(0.05,0.4)#(0.05,0.2)
    b_A_over_b_I = np.random.uniform(0,1)
    
    private_activity_SEA = np.random.uniform(0,1)**(1/2)#change of private behavior for ppl in SEA: 1=no change in behavior, 0=all contacts stopped 
    public_activity_SEA = np.random.uniform(0,1)**(1/2)#change of public behavior for ppl in SEA: 1=no change in behavior, 0=all contacts stopped 
    activity_reduction_I = np.random.uniform(0,1) #change of public behavior for ppl in I: 0=no change in behavior, 1=all contacts stopped 
    activity_reduction_P = np.random.uniform(0.8,1)#change of public behavior for ppl in P (tested positive): 0=no change in behavior, 1=all contacts stopped 
    activity_reduction_old = 1-np.random.uniform(0,1)**(1/2)#change of public behavior for old ppl: 0=no change in behavior, 1=all contacts stopped 
    
    p_A = np.random.uniform(0.05,0.5)
    p_A_young_over_p_A_old = np.random.uniform(1,5)
    p_IH_old_over_p_IH_young = np.random.uniform(4,10)
    p_HD_old_over_p_HD_young = np.random.uniform(4,10)
    
    triaging_policy = np.random.choice(['FCFS','same','FBLS','FBrnd'])
    max_number_of_tests_available_per_day = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,20,40])
    testing_policy_decision_major = np.random.choice(['O>Y','Y>O'])
    testing_policy_decision_minor = np.random.choice(['random','FIFT','LIFT'])
    testing_delay = np.random.randint(0,8)
    
    for k in [4,6,10]:
        #derived/fitted parameters
        b_A = b_A_over_b_I*b_I
        b_H = b_I
        p_IH_young_over_p_IH_old = 1/p_IH_old_over_p_IH_young
        p_HD_young_over_p_HD_old = 1/p_HD_old_over_p_HD_young
        probability_of_close_contact_with_random_person_in_public = k/(N-1)
        private_activity_I = private_activity_SEA*(1-activity_reduction_I)#change of private behavior for ppl in I: 1=no change in behavior, 0=all contacts stopped 
        public_activity_I = public_activity_SEA*(1-activity_reduction_I)#change of public behavior for ppl in I: 1=no change in behavior, 0=all contacts stopped 
        private_activity_H = private_activity_SEA*(1-activity_reduction_H)#change of private behavior for ppl in H: 1=no change in behavior, 0=all contacts stopped 
        public_activity_H = public_activity_SEA*(1-activity_reduction_H)#change of public behavior for ppl in H: 1=no change in behavior, 0=all contacts stopped 
        activity_P = 1-activity_reduction_P#change of public behavior for ppl in P (tested positive): 1=no change in behavior, 0=all contacts stopped 
        activity_old = 1-activity_reduction_old#change of public behavior for old ppl: 1=no change in behavior, 0=all contacts stopped 
    
        neverinfected_count,_,_,_,death_count,seed,R0,mean_generation_time,mean_time_infection,infections_caused_byE,infections_caused_byA,infections_caused_byI,total_initial_edge_weight,max_len_H,max_len_I,max_len_not_tested = model(N,k,p_edgechange,network_generating_function,p_old,T_max,b_A,b_I,b_H,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_young_over_p_A_old,p_IH,p_IH_young_over_p_IH_old,overall_death_rate_covid19,p_HD_young_over_p_HD_old,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,private_activity_I,private_activity_H,private_old_young_activity,public_activity_SEA,activity_old,public_activity_I,public_activity_H,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,seed=None,OUTPUT=False,ESTIMATE_R0=True,interpolator_per_day_death_rate=interpolator_per_day_death_rate)
        neverinfected_counts.append(neverinfected_count)
        death_counts.append(death_count)
        R0s.append(R0)
        mean_generation_times.append(mean_generation_time)
        mean_time_infections.append(mean_time_infection)
        infections_caused_byEs.append(infections_caused_byE)
        infections_caused_byAs.append(infections_caused_byA)
        infections_caused_byIs.append(infections_caused_byI)
        total_initial_edge_weights.append(total_initial_edge_weight)
        max_len_Hs.append(max_len_H)
        max_len_Is.append(max_len_I)
        max_len_not_testeds.append(max_len_not_tested)
        
        args.append([N,k,p_edgechange,network_generating_function,p_old,T_max,b_I,b_A_over_b_I,params_EA,params_EI,params_IH,params_AR,params_IR,params_HR,p_A,p_A_young_over_p_A_old,p_IH,p_IH_young_over_p_IH_old,overall_death_rate_covid19,p_HD_young_over_p_HD_old,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,public_activity_SEA,activity_reduction_I,activity_reduction_H,activity_reduction_old,private_old_young_activity,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_reduction_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,seed])
args = np.array(args)
#to get args_names, run this ','.join(["'"+el+"'" for el in '''N,k,p_edgechange,network_generating_function,p_old,T_max,b_I,b_A_over_b_I,t_EA,t_EI,t_IH,t_AR,t_IR,t_HR,p_A,p_A_young_over_p_A_old,p_IH,p_IH_young_over_p_IH_old,overall_death_rate_covid19,p_HD_young_over_p_HD_old,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,public_activity_SEA,activity_reduction_I,activity_reduction_H,private_old_young_activity,hospital_beds_per_person,care_decline_exponent,triaging_policy,max_number_of_tests_available_per_day,activity_reduction_P,testing_policy_decision_major,testing_policy_decision_minor,testing_delay,initial_seed'''.split(',')])
args_names = ['N','k','p_edgechange','network_generating_function','p_old','T_max','b_I','b_A_over_b_I','params_EA','params_EI','params_IH','params_AR','params_IR','params_HR','p_A','p_A_young_over_p_A_old','p_IH','p_IH_young_over_p_IH_old','overall_death_rate_covid19','p_HD_young_over_p_HD_old','probability_of_close_contact_with_random_person_in_public','private_activity_SEA','public_activity_SEA','activity_reduction_I','activity_reduction_H','activity_reduction_old','private_old_young_activity','hospital_beds_per_person','care_decline_exponent','triaging_policy','max_number_of_tests_available_per_day','activity_reduction_P','testing_policy_decision_major','testing_policy_decision_minor','testing_delay','initial_seed']

f = open(output_folder+'output_model%s_nsim%i_N%i_kvariable_seed%i_SLURM_ID%i.txt' % (version,nsim,N,initial_seed,SLURM_ID) ,'w')
f.write('filename\t'+filename+'\n')
f.write('SLURM_ID\t'+str(SLURM_ID)+'\n')
for ii in range(len(args_names)):#enumerate(zip(args,args_names)):
    if args_names[ii] in ['network_generating_function'] or len(set(args[:,ii])) == 1:
        f.write(args_names[ii]+'\t'+str(args[0,ii])+'\n')    
    else:
        f.write(args_names[ii]+'\t'+'\t'.join(list(map(str,[el if type(el)!=float else round(el,6) for el in args[:,ii]])))+'\n')

f.write('neverinfected_counts\t'+'\t'.join(list(map(str,neverinfected_counts)))+'\n')
f.write('death_counts\t'+'\t'.join(list(map(str,death_counts)))+'\n')
f.write('R0s\t'+'\t'.join(list(map(str,[round(el,3) for el in R0s])))+'\n')
f.write('mean_generation_times\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_generation_times])))+'\n')
f.write('mean_time_infections\t'+'\t'.join(list(map(str,[round(el,3) for el in mean_time_infections])))+'\n')
f.write('infections_caused_byEs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byEs])))+'\n')
f.write('infections_caused_byAs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byAs])))+'\n')
f.write('infections_caused_byIs\t'+'\t'.join(list(map(str,[round(el,3) for el in infections_caused_byIs])))+'\n')
f.write('total_initial_edge_weights\t'+'\t'.join(list(map(str,[round(el,6) for el in total_initial_edge_weights])))+'\n')
f.write('max_len_Hs\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_Hs])))+'\n')
f.write('max_len_Is\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_Is])))+'\n')
f.write('max_len_not_testeds\t'+'\t'.join(list(map(str,[round(el,3) for el in max_len_not_testeds])))+'\n')
f.close()


#for debugging
#public_activity_I=1
#private_activity_I=1
#
#b_A=0.1
#b_I=0.2
#b_H=0.2
#seed = None
#
#triaging_policies = ['FCFS','same','LSF','FBLS']
#ys=[]
#nsim=100
#for xf in triaging_policies:
#    ys.append([model(N,k,p_edgechange,network_generating_function,p_old,T_max,p_A,p_A_young_over_p_A_old,b_A,b_I,b_H,t_EA,t_EI,t_IH,t_AR,t_IR,t_HR,p_IH_young,p_IH_old,p_HD_young,p_HD_old,per_day_death_rate_hospital,probability_of_close_contact_with_random_person_in_public,private_activity_SEA,private_activity_I,private_activity_H,private_old_young_activity,public_activity_SEA,public_activity_I,public_activity_H,hospital_beds_per_person,care_decline_exponent,triaging_policy,seed=seed,OUTPUT=False,ESTIMATE_R0=False)[3] for _ in range(nsim)])


#f,ax = plt.subplots()
#b = 0.05
#t_EI = 5
#x = np.arange(0,t_EI,0.01)
#for multiplier in [1,0.1,0.01,0.001,0.00001,1e-10]:
#    f0 = b*multiplier
#    ax.plot(x,f0*np.exp(np.log(b/f0)/t_EI*x),label=str(multiplier))
#ax.legend(loc='best')
#ax.set_xlabel('time')
#ax.set_ylabel('infectivity')
#plt.savefig('parameter_choice_for_infectivity_during_E.pdf')
#
#f,ax = plt.subplots()
#b = 0.05
#t_EI = 5
#x = np.arange(0,t_EI,0.01)
#for multiplier in [1,0.1,0.01,0.001,0.00001,1e-10]:
#    f0 = b*multiplier
#    ax.plot(x,f0*np.exp(np.log(b/f0)/t_EI*x)-f0+multiplier*b/t_EI*x,label=str(multiplier))
#ax.legend(loc='best')
#ax.set_xlabel('time')
#ax.set_ylabel('infectivity')
#plt.savefig('parameter_choice_for_infectivity_during_E_linear_addition.pdf')


#exponent_old = 1
#exponent_SEA = 1/2
#_,_,_,n_oo,n_oy,n_yy = total_edge_weight_in_susceptible_network(network,ages,activity_old,private_activity_SEA,public_activity_SEA,c)
#res = []
#for i in range(1000):
#    activity_old = np.random.random()**exponent_old
#    private_activity_SEA = np.random.random()**exponent_SEA
#    public_activity_SEA = np.random.random()**exponent_SEA
#    total_edge_weight = total_edge_weight_in_susceptible_network(network,ages,activity_old,private_activity_SEA,public_activity_SEA,c,n_oo,n_oy,n_yy)[0]
#    res.append(total_edge_weight)
#plt.hist(res,bins=50)
#
#
#
#for b in [0.05,0.1,0.2]:
#    res = []
#    for _ in range(100):
#        val=1
#        T = np.random.poisson(5)
#        f0=0.001
#    
#        for t in range(1,T+1):
#            infectivity = t/T*b
#            val *= 1 - infectivity
#        res.append(1-val)
#    print('linear','b =',b,':',np.round(np.mean(res),3))
#
#for m in [0.001,0.01,0.1,1]:
#    for b in [0.05,0.1,0.2]:
#        res = []
#        for _ in range(100):
#            val=1
#            T = np.random.poisson(5)
#            
#        
#            for t in range(1,T+1):
#                infectivity = b*m*np.exp(np.log(1/m)*t/T)-b*m+b*m*t/T
#                #infectivity = t/T*b
#                val *= 1 - infectivity
#            res.append(1-val)
#        print('exp with m = %s,' % str(m),'b =',b,':',np.round(np.mean(res),3))
#
#
#
#
##look at gamma distributions
#nsim = 10000
#k = 1.7
#sigma = 3.33
##k = 10
##sigma = 0.5
#mean = k*sigma
#variance = k*sigma**2
#mean2 =k*k*sigma**2
#variance_divided_by_mean2 = 1/k
#
#
#data = np.random.gamma(k,sigma,nsim)
##sorted_data = np.sort(data)
#f,ax = plt.subplots()
##ax.step(sorted_data, np.arange(sorted_data.size))  # From 0 to the number of data points-1
#ax.hist(data,bins=50)
#
#k = 2#1.7
#sigma = 2#3.4
#x=np.arange(0,20,0.01)
#y = x**(k-1)*np.exp(-x/sigma)
#xx = np.arange(0,20)
#yy = xx**(k-1)*np.exp(-xx/sigma)
#f,ax = plt.subplots()
#ax.plot(x,y/max(y)*0.12)
#ax.plot(xx,yy/max(y)*0.12,'rx')
#
#
#
#
#
#
###model analysis
#f,ax=plt.subplots()
#time_infected = np.array(time_infected)
#counter_secondary_infections = np.array(counter_secondary_infections)
#x = time_infected[time_infected>=0]
#y = counter_secondary_infections[time_infected>=0]
#[x,y] = np.array(sorted(zip(x,y))).T
#R0_running = [a/b for a,b in zip(np.cumsum(y),np.arange(1,len(x)+1))]
#ax.plot(x,R0_running,label='running R0')
#
#disease_generation_time = np.array(disease_generation_time)
#x = time_infected[time_infected>0]
#y = disease_generation_time[time_infected>0]
#[x,y] = np.array(sorted(zip(x,y))).T
#disease_generation_time_running = [a/b for a,b in zip(np.cumsum(y),np.arange(1,len(x)+1))]
#ax.plot(x,disease_generation_time_running,label='disease_generation_time running')
#
#ax.legend(loc='best')
#plt.savefig('example_run_runningR0.pdf')
#
#
#f,ax=plt.subplots()
#ax.scatter(mean_time_infections,R0s,c=neverinfected_counts)
#ax.set_xlabel('time 50% infected')
#ax.set_ylabel('Initial R0')
#plt.savefig('R0s_vs_mean_time_infections.pdf')
#
#f,ax=plt.subplots()
#ax.scatter(mean_generation_times,R0s,c=neverinfected_counts)
#ax.set_xlabel('mean_generation_time')
#ax.set_ylabel('Initial R0')
#plt.savefig('R0s_vs_mean_generation_time.pdf')
#
#f,ax=plt.subplots()
#ax.scatter(mean_generation_times,mean_time_infections,c=neverinfected_counts)
#ax.set_xlabel('mean_generation_time')
#ax.set_ylabel('time 50% infected')
#plt.savefig('mean_time_infections_vs_mean_generation_time.pdf')

#Table 1, Case Report 9, find death rates of non critical care patients
pI = 2/3

IFRs = np.array([0.002,0.006,0.03,0.08,0.15,0.6,2.2,5.1,9.3])/100
pIHs = np.array([0.1,0.3,1.2,3.2,4.9,10.2,16.6,24.3,27.3])/100
pHCs = np.array([5,5,5,5,6.3,12.2,27.4,43.2,70.9])/100

res= []
for IFR,pIH,pHC in zip(IFRs,pIHs,pHCs):
    pHD = (IFR/(pI*pIH)-pHC*0.5)/(1-pHC)
    res.append([IFR,pIH,pHC,pHD,pHC*0.5+(1-pHC)*pHD])
res = np.round(np.array(res)*100,2)

#https://www.census.gov/data/datasets/2017/demo/popproj/2017-popproj.html
a='''4113164	4110117	4104058	4094281	4016919	4039164	4033531	4022626	4029209	4075879	4074733	4076315	4205008	4223909	4190013	4180890	4197186	4179617	4180135	4300527	4384632	4338146	4359234	4386967	4430497	4529025	4619893	4685941	4800614	4876514	4856771	4660832	4542808	4455998	4463007	4485228	4337294	4392096	4381171	4326220	4400565	4126454	4039827	3992089	3870851	3982690	3854262	3905661	4071128	4290890	4332660	4099623	4005824	4002980	4069319	4303838	4373935	4362272	4344637	4387070	4411539	4251390	4212048	4149974	3985841	3934139	3755158	3594166	3435726	3310982	3211395	3093441	3033884	3130130	2290416	2233196	2145506	2160437	1852886	1661013	1537134	1410262	1312397	1171078	1077340	990131	847852	778518	701025	625755	558560	464985	398712	330389	264318	212880	163348	121128	88491	62724	92064'''
a=list(map(int,a.split('\t')))
nr_ppl = []
for i in range(8):
    nr_ppl.append(sum(a[10*i:10*(i+1)]))
nr_ppl.append(sum(a[10*8:]))
nr_ppl = np.array(nr_ppl)
weights = nr_ppl/sum(nr_ppl)
print('high-risk vs low-risk hospitalized if symptomatic',np.dot(res[:-3,1],weights[:-3])/sum(weights[:-3]),np.dot(res[-3:,1],weights[-3:])/sum(weights[-3:]))
print('high-risk vs low-risk dying if hospitalized',np.dot(res[:-3,4],weights[:-3])/sum(weights[:-3]),np.dot(res[-3:,4],weights[-3:])/sum(weights[-3:]))


