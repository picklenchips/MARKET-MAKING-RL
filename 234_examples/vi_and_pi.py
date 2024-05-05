### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

# remove graphing functionality for autograder

from cycler import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

hexcolors = ['DC267F', '648FFF', '785EF0', 'FE6100', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])


np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    ############################
    # YOUR IMPLEMENTATION HERE #
    state = int(state); action = int(action)
    backup_val = R[state, action] # immediate reward
    backup_val += gamma * np.dot(T[state,action,:],V) # discounted reward
    ############################

    return backup_val

# (DETERMINISTIC - ideally, policy will be a num_states x num-states array)
def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    error = 1
    while error > tol:
        max_error = 0
        new_value_function = np.zeros(num_states)
        for state in range(num_states):
            action = policy[state]
            new_val = bellman_backup(state, action, R, T, gamma, value_function)
            # compute new error
            new_error = abs(value_function[state] - new_val)
            if new_error > max_error:
                max_error = new_error
            # this would converge faster, no?
            # value_function[state] = new_val
            new_value_function[state] = new_val
        value_function = new_value_function
        error = max_error
    ############################
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for state in range(num_states):
        best_val = 0
        for action in range(num_actions):
            new_val = bellman_backup(state, action, R, T, gamma, V_policy)
            if new_val > best_val:
                new_policy[state] = action
                best_val = new_val
    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    V_policy = policy_evaluation(policy,R,T,gamma,tol)
    error = 1
    while error:
        old_policy = policy
        policy = policy_improvement(policy, R, T, V_policy, gamma)
        new_V = policy_evaluation(policy, R, T, gamma, tol)
        error = np.max(np.abs(policy - old_policy))
        V_policy = new_V
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    error = 1
    while error > tol:
        max_diff = 0
        new_policy = np.zeros(num_states, dtype=int)
        new_value_function = np.zeros(num_states)
        for state in range(num_states):
            old_val = value_function[state]
            for action in range(num_actions):
                new_val = bellman_backup(state, action, R, T, gamma, value_function)
                if new_val > new_value_function[state]: # we have a good action
                    new_policy[state] = action
                    new_value_function[state] = new_val
            if new_value_function[state] != old_val:
                new_diff = abs(old_val - new_value_function[state])
                if new_diff > max_diff:
                    max_diff = new_diff
        policy = new_policy
        value_function = new_value_function
        error = max_diff
    ############################
    return value_function, policy

def print_results(func, gamma, R, T, tol):
    print("gamma: %.15f" % gamma)
    value, policy = func(R, T, gamma, tol)
    print("\t",value)
    print([['L', 'R'][a] for a in policy])

def print_exact_gammas(possible_currents, gamma_tol=1e-15):
    for current in possible_currents:
        print(f"\ngetting gamma for {current} current:",end=' ')
        env = RiverSwim(current, SEED)
        R, T = env.get_model()
        
        for i_func in [value_iteration, policy_iteration]:
            gamma       = 0.99
            value, policy = i_func(R, T, gamma, tol)
            delta_gamma = 1
            
            while delta_gamma > gamma_tol:
                delta_gamma /= 10
                
                while policy[0]:
                    gamma -= delta_gamma
                    if gamma <= 0: break
                    value, policy = i_func(R, T, gamma, tol)
                gamma += delta_gamma
                policy[0] = 1
            gamma -= delta_gamma
            print(i_func.__name__,":")
            # confirm that this is indeed the boundary between swimming 
            # right or left at first current
            for i in range(2):
                gamma = gamma + delta_gamma*i
                print_results(i_func, gamma, R, T, tol)


def plot_gammas(possible_currents, gamma_tol=1e-2, separate=True):
    if separate:
        fig, axs = plt.subplots(1,3,figsize=(12,4),sharey=True)
        plt.subplots_adjust(wspace=0.05,left=0.05,right=0.99)
        for i, ax in enumerate(axs):
            ax.set_yscale('log')
            ax.invert_xaxis()
            ax.set(xlabel=r"$\gamma$")
            if not i:
                ax.set(ylabel='state values')
            ax.set(title=possible_currents[i])
    else:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.invert_xaxis()
        ax.set(xlabel=r"$\gamma$", ylabel='state values',yscale='log')
    for idx,current in enumerate(possible_currents):
        if separate:
            ax = axs[idx]
        print(f"\ngetting gamma for {current} current:",end=' ')
        env = RiverSwim(current, SEED)
        R, T = env.get_model()
        gamma = 0.99
        for i_func in [value_iteration]:
            print(i_func.__name__,":")
            all_values = []
            gammas = []
            gamma = 0.99
            value, policy = i_func(R, T, gamma, tol)
            all_values.append(value)
            gammas.append(gamma)
            while policy[0]:
                gamma -= gamma_tol
                if gamma <= 0: break
                value, policy = i_func(R, T, gamma, tol)
                all_values.append(value)
                gammas.append(gamma)
            all_values = np.array(all_values)
            print((all_values[-1]-all_values[-2])/gamma_tol)
            for i in range(all_values.shape[1]):
                if not separate:
                    if not i:
                        ax.plot(gammas, all_values[:,i], label=current,c=f'C{idx}')
                    else:
                        ax.plot(gammas, all_values[:,i], c=f'C{idx}')
                else:
                    ax.plot(gammas, all_values[:,i])
            #ax.set(title=current)
    if not separate:
        plt.legend()
    name = "currents"
    if not separate: name += "1"
    fig.savefig(f"../{name}.pdf",bbox_inches="tight")
    print(f'saved fig to {name}.pdf')
    plt.show()


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    current_index = 0
    discount_factor = 0.99
    tol=1e-8

    # should get 30.328, 36.859
    # WEAK - 0.0122
    # MEDI - 0.0122
    # STRO - 0.0097

    possible_currents = ['WEAK', 'MEDIUM', 'STRONG']
    RIVER_CURRENT = possible_currents[current_index]
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=tol)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=tol)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])

    i_current = 0
    #print_exact_gammas(possible_currents, gamma_tol=1e-2)
    plot_gammas(possible_currents, gamma_tol=1e-3)



        
