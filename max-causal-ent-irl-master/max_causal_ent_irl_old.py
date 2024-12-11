import numpy as np 
from frozen_lake import FrozenLakeEnv, EthanCommute, cautious_ethan, don_care_ethan, ethan_belief
from mdps import MDP, MDPOneTimeR
from traj_tools import generate_trajectories, compute_s_a_visitations
from value_iter_and_policy import vi_boltzmann, vi_rational, qi_boltzmann, opt_boltzmann
from occupancy_measure import compute_D
from matplotlib import pyplot as plt

import torch

def max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma=1, h=None, 
                       temperature=1, epochs=1, learning_rate=0.2, theta=None, q_opt = None, v_opt = None):
    '''
    Finds theta, a reward parametrization vector (r[s] = features[s]'.*theta) 
    that maximizes the log likelihood of the given expert trajectories, 
    modelling the expert as a Boltzmann rational agent with given temperature. 
    
    This is equivalent to finding a reward parametrization vector giving rise 
    to a reward vector giving rise to Boltzmann rational policy whose expected 
    feature count matches the average feature count of the given expert 
    trajectories (Levine et al, supplement to the GPIRL paper).

    Parameters
    ----------
    mdp : object
        Instance of the MDP class.
    feature_matrix : 2D numpy array
        Each of the rows of the feature matrix is a vector of features of the 
        corresponding state of the MDP. 
    trajectories : 3D numpy array
        Expert trajectories. 
        Dimensions: [number of traj, timesteps in the traj, state and action].
    gamma : float 
        Discount factor; 0<=gamma<=1.
    h : int
        Horizon for the finite horizon version of value iteration.
    temperature : float >= 0
        The temperature parameter for computing V, Q and policy of the 
        Boltzmann rational agent: p(a|s) is proportional to exp(Q/temperature);
        the closer temperature is to 0 the more rational the agent is.
    epochs : int
        Number of iterations gradient descent will run.
    learning_rate : float
        Learning rate for gradient descent.
    theta : 1D numpy array
        Initial reward function parameters vector with the length equal to the 
        #features.
    Returns
    -------
    1D numpy array
        Reward function parameters computed with Maximum Causal Entropy 
        algorithm from the expert trajectories.
    '''    
    
    # Compute the state-action visitation counts and the probability 
    # of a trajectory starting in state s from the expert trajectories.
    sa_visit_count, P_0 = compute_s_a_visitations(mdp, gamma, trajectories)
    sa_visit_count = torch.tensor(sa_visit_count)
    print("optimal", -torch.mean(sa_visit_count * (q_opt - v_opt)))
    
    # Mean state visitation count of expert trajectories
    # mean_s_visit_count[s] = ( \sum_{i,t} 1_{traj_s_{i,t} = s}) / num_traj
    #mean_s_visit_count = np.sum(sa_visit_count,1) / trajectories.shape[0]
    # Mean feature count of expert trajectories
    #mean_f_count = np.dot(feature_matrix.T, mean_s_visit_count)
    
    if theta is None:
        #theta = np.eye(513)*513
        theta = np.random.rand(feature_matrix.shape[1], feature_matrix.shape[0])
        
    q_opt = torch.tensor(q_opt).double()
    v_opt = torch.tensor(v_opt).double()
    theta = torch.tensor(theta, requires_grad=True)
    feature_matrix = torch.tensor(feature_matrix).double()
    optim = torch.optim.Adam([theta], learning_rate)
        
    losses = []
    for i in range(epochs):
        belief = torch.matmul(feature_matrix, theta)
        belief = torch.nn.functional.softmax(belief/.5, dim=1)
        #belief = torch.nn.functional.normalize(belief, 1, 1)
        #belief /= torch.sum(belief, dim = 1)
        
        V = torch.matmul(belief, v_opt)
        Q = torch.matmul(belief, q_opt)
        
        
        
        #print(r.shape)
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        #V,Q,policy = opt_boltzmann(mdp, gamma, r, q_opt, v_opt, h, temperature)
        #V, Q, policy = vi_boltzmann(mdp, gamma, r, h, temperature)
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = -torch.sum(sa_visit_count * (Q - V))
        optim.zero_grad()
        L.backward()
        optim.step()
        
        # The expected #times policy π visits state s in a given #timesteps.
        #D = compute_D(mdp, gamma, policy, P_0, t_max=trajectories.shape[1])    
        #print(D.shape)    

        # IRL log likelihood gradient w.r.t rewardparameters. 
        # Corresponds to line 9 of Algorithm 2 from the MaxCausalEnt IRL paper 
        # www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf. 
        # Negate to get the gradient of neg log likelihood, 
        # which is then minimized with GD.
        #dL_dtheta = -(mean_f_count - np.dot(feature_matrix.T, D))

        # Gradient descent
        #theta = theta - learning_rate * dL_dtheta

        if (i+1)%10==0: 
            print('Epoch: {} log likelihood of all traj: {}'.format(i,L), 
                  ', average per traj step: {}'.format(
                  L/(trajectories.shape[0] * trajectories.shape[1])), end = "\r")
            losses.append(-L.item())
    return theta, losses, V, Q


def main(t_expert=1e-2,
         t_irl=1e-2,
         gamma=1,
         h=10,
         n_traj=200,
         traj_len=10,
         learning_rate=0.01,
         epochs=300):
    '''
    Demonstrates the usage of the implemented MaxCausalEnt IRL algorithm. 
    
    First a number of expert trajectories is generated using the true reward 
    giving rise to the Boltzmann rational expert policy with temperature t_exp. 
    
    Hereafter the max_causal_ent_irl() function is used to find a reward vector
    that maximizes the log likelihood of the generated expert trajectories, 
    modelling the expert as a Boltzmann rational agent with temperature t_irl.
    
    Parameters
    ----------
    t_expert : float >= 0
        The temperature parameter for computing V, Q and policy of the 
        Boltzmann rational expert: p(a|s) is proportional to exp(Q/t_expert);
        the closer temperature is to 0 the more rational the expert is.
    t_irl : float
        Temperature of the Boltzmann rational policy the IRL algorithm assumes
        the expert followed when generating the trajectories.
    gamma : float 
        Discount factor; 0<=gamma<=1.
    h : int
        Horizon for the finite horizon version of value iteration subroutine of
        MaxCausalEnt IRL algorithm.
    n_traj : int
        Number of expert trajectories generated.
    traj_len : int
        Number of timesteps in each of the expert trajectories.
    learning_rate : float
        Learning rate for gradient descent in the MaxCausalEnt IRL algorithm.
    epochs : int
        Number of gradient descent steps in the MaxCausalEnt IRL algorithm.
    '''
    np.random.seed(0)
    #mdp = MDPOneTimeR(FrozenLakeEnv(is_slippery=False))  
    task = EthanCommute()
    r_expert = task.rmat  
    mdp = MDP(task)
    #print(mdp.T.shape)

    # Features
    feature_matrix = np.eye(mdp.nS)
    #feature_matrix = np.array(task.states)
    #for row in feature_matrix:
    #    if row[0] == 1:
    #        row[6:] = 0
    #print("fm", feature_matrix)
    #print(feature_matrix.shape)
    
    # Add dummy feature to show that features work
    if False:
        feature_matrix = np.concatenate((feature_matrix, np.ones((mdp.nS,1))), 
                                        axis=1)
    
    # The true reward weights and the reward
    #theta_expert = np.zeros(feature_matrix.shape[1])
    #theta_expert[24] = 1
    #r_expert = np.dot(feature_matrix, theta_expert)
    
    # Compute the Boltzmann rational expert policy from the given true reward.
    if t_expert>0:
        V, Q, policy_expert = qi_boltzmann(mdp, gamma, r_expert, h, t_expert)
    if t_expert==0:
        V, Q, policy_expert = vi_rational(mdp, gamma, r_expert, h)
        
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    print(compute_s_a_visitations(mdp, gamma, trajectories)[0])
    #print(Q-V)
    #print(policy_expert)
    #print(policy_expert)
    
    belief = ethan_belief(task)
    
    counts = {"clear": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "cloudy": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "raining": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "windy": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}}
    
    # ts = task.text_states
    # for i in range(len(ts)):
    #     for j in range(len(ts)):
    #         if "done" in ts[i] or "done" in ts[j]: continue
    #         if "home" in ts[i]:
    #             counts[ts[i][1]][ts[j][2]] += belief[i][j].item()
    #             counts[ts[i][1]][ts[j][3]] += belief[i][j].item()
    #         elif "work" in ts[i]:
    #             counts[ts[i][2]][ts[j][3]] += belief[i][j].item()
    # print(counts)
    
    # for key in counts.keys():
    #     total = 0
    #     for keys in counts[key].keys():
    #         total += counts[key][keys]
    #     for keys in counts[key].keys():
    #         counts[key][keys] /= total
    #     print(key, counts[key])
    
    #print(belief)
    V_pol = np.matmul(belief, V)
    Q_pol = np.matmul(belief, Q)
    
    #print((Q_pol - V_pol) - (Q-V))
    #expt = lambda x: np.exp(x/t_expert)
    policy_expert = torch.nn.functional.softmax(torch.tensor(Q_pol - V_pol)/t_expert).numpy()
    #policy_expert = expt(V_pol - Q_pol)
    # for i in range(len(task.text_states)):
    #     print(task.text_states[i], policy_expert[i])
    
    #policy_expert = don_care_ethan(task)
    #policy_expert = torch.nn.functional.softmax(torch.tensor(policy_expert)/t_expert)
    #policy_expert = policy_expert.numpy()
    #print(policy_expert)
    
    
        
    # Generate expert trajectories using the given expert policy.
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    #print(trajectories)
    
    # Compute and print the stats of the generated expert trajectories.
    sa_visit_count, _ = compute_s_a_visitations(mdp, gamma, trajectories)
    print(sa_visit_count)
    log_likelihood = np.sum(sa_visit_count * (Q_pol - V_pol).numpy())
    
    print('Generated {} traj of length {}'.format(n_traj, traj_len))
    print('Log likelihood of all traj under the policy generated ', 
          'from the true reward: {}, \n average per traj step: {}'.format(
           log_likelihood, log_likelihood / (n_traj * traj_len)))
    #print('Average return per expert trajectory: {} \n'.format(
    #        np.sum(np.sum(sa_visit_count, axis=1)*( np.sum(policy_expert*Q, axis = 1) )) / n_traj))

    # Find a reward vector that maximizes the log likelihood of the generated 
    # expert trajectories.
    
    print(Q_pol)
    
    theta, losses, V, Q = max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma, h, 
                               t_irl, epochs, learning_rate, q_opt = Q, v_opt = V)
    print('Final reward weights: ', theta)
    feature_matrix = torch.tensor(feature_matrix).double()
    probs = torch.nn.functional.softmax( torch.matmul(feature_matrix, theta) )#.argmax(dim=1)
    
    
    for i in range(len(task.text_states)):
        for j in range(len(task.text_states)):
            if task.text_states[i][0] != task.text_states[j][0]:
                probs[i,j] = 0
    
    vals = probs.argmax(dim = 1)
    
    with torch.no_grad():
        temperature = t_irl
        expt = lambda x: torch.exp(x/temperature)
        policy = expt(Q - V).numpy()
    
    counts = {"clear": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "cloudy": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "raining": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}
              , "windy": {"clear": 0, "cloudy": 0, "raining": 0, "windy": 0}}
    
    ts = task.text_states
    for i in range(len(ts)):
        for j in range(len(ts)):
            if "done" in ts[i] or "done" in ts[j]: continue
            if "home" in ts[i]:
                counts[ts[i][1]][ts[j][2]] += probs[i][j].item()
                counts[ts[i][1]][ts[j][3]] += probs[i][j].item()
            elif "work" in ts[i]:
                counts[ts[i][2]][ts[j][3]] += probs[i][j].item()
    print(counts)
    
    for key in counts.keys():
        total = 0
        for keys in counts[key].keys():
            total += counts[key][keys]
        for keys in counts[key].keys():
            counts[key][keys] /= total
        print(key, counts[key])
        
        
    
    
    for i in range(len(task.text_states)):
       print(task.text_states[i], task.text_states[vals[i]])
    #for i in range(len(task.text_states)):
    #    print(task.text_states[i], policy[i], policy_expert[i])
    
    plt.figure(dpi = 150)
    plt.plot(losses, label = "Predicted Reward")
    #plt.plot([0, len(losses)], [-24.66787025854367, -24.66787025854367], label = "True Reward Function")
    plt.xlabel("Optimization Iterations")
    plt.ylabel("Total Log Likelihood")
    plt.legend()
    plt.title("Log Likelihood of Trajectories over Time")
    plt.savefig("maxcent.png")

if __name__ == "__main__":
    main(traj_len=3, n_traj= 100_000, epochs = 20_000, learning_rate= .01)