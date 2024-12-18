import numpy as np 
from frozen_lake import *
from mdps import MDP, MDPOneTimeR
from traj_tools import generate_trajectories, compute_s_a_visitations
from value_iter_and_policy import vi_boltzmann, torch_boltzmann, vi_rational, qi_boltzmann
from occupancy_measure import compute_D
from matplotlib import pyplot as plt

import torch

import warnings
warnings.filterwarnings("ignore")

def max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma=1, h=None, 
                       temperature=1, epochs=1, learning_rate=0.2, theta=None, q_opt = None, v_opt = None, r = None):
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
    #print("optimal", -torch.mean(sa_visit_count * (q_opt - v_opt)))
    
    # Mean state visitation count of expert trajectories
    # mean_s_visit_count[s] = ( \sum_{i,t} 1_{traj_s_{i,t} = s}) / num_traj
    #mean_s_visit_count = np.sum(sa_visit_count,1) / trajectories.shape[0]
    # Mean feature count of expert trajectories
    #mean_f_count = np.dot(feature_matrix.T, mean_s_visit_count)
    
    if theta is None:
        #theta = np.eye(513)*513
        theta = np.array([[.5, .5], [.5, .5]])
        #theta = np.random.rand(feature_matrix.shape[1], feature_matrix.shape[0])
        
    q_opt = torch.tensor(q_opt).double()
    v_opt = torch.tensor(v_opt).double()
    r = torch.tensor(r).double()
    theta = torch.tensor(theta, requires_grad=True)
    feature_matrix = torch.tensor(feature_matrix).double()
    optim = torch.optim.Adam([theta], learning_rate)
        
    losses = []
    for i in range(epochs):
        belief = torch.matmul(feature_matrix, theta)
        
        #belief /= np.sum(belief, axis = 0)
        belief = torch.nn.functional.softmax(belief, dim=1)
        #belief = torch.nn.functional.normalize(belief, 1, 1)
        #belief /= torch.sum(belief, dim = 1)
        
        #V = torch.matmul(belief, v_opt)
        #Q = torch.matmul(belief, q_opt)
        
        
        
        #print(r.shape)
        # Compute the Boltzmann rational policy \pi_{s,a} = \exp(Q_{s,a} - V_s) 
        #V,Q,policy = opt_boltzmann(mdp, gamma, r, q_opt, v_opt, h, temperature)
        V, Q, policy = torch_boltzmann(mdp, gamma, torch.matmul(belief, r), h, temperature)
        
        # IRL log likelihood term: 
        # L = 0; for all traj: for all (s, a) in traj: L += Q[s,a] - V[s]
        L = -torch.sum(sa_visit_count * (Q - V))
        
        #policy = torch.nn.functional.softmax(torch.tensor(Q - V)/temperature).numpy()
        
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
        #dL_dtheta = -dL_dtheta*np.eye(2)
        #for row in dL_dtheta:
        #    row = 0

        # Gradient descent
        #theta = theta - learning_rate * dL_dtheta

        if (i+1)%10==0: 
            print('Epoch: {} log likelihood of all traj: {}'.format(i,L), 
                  ', average per traj step: {}'.format(
                  L/(trajectories.shape[0] * trajectories.shape[1])), end = "\r")
            losses.append(-L.item())
    print()
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
    task = BinGame()
    r_expert = np.array([1,0])
    #r_expert = task.rmat  
    mdp = MDP(task)
    #print(mdp.T.shape)

    # Features
    feature_matrix = np.eye(mdp.nS)

    belief = np.array([[.8, .2], [.2, .8]])
    print(belief)
    
    # Compute the Boltzmann rational expert policy from the given true reward.
    if t_expert>0:
        V, Q, policy_expert = vi_boltzmann(mdp, gamma, np.matmul(belief, r_expert), h, t_expert)
    if t_expert==0:
        V, Q, policy_expert = vi_rational(mdp, gamma, r_expert, h)
    print(policy_expert)
    print(Q)
    print(V)
        
    # Generate expert trajectories using the given expert policy.
    trajectories = generate_trajectories(mdp, policy_expert, traj_len, n_traj)
    #print(trajectories)
    
    # Compute and print the stats of the generated expert trajectories.
    sa_visit_count, _ = compute_s_a_visitations(mdp, gamma, trajectories)
    print(sa_visit_count)
            
    
    log_likelihood = np.sum(sa_visit_count * (Q - V))
    
    print('Generated {} traj of length {}'.format(n_traj, traj_len))
    print('Log likelihood of all traj under the policy generated ', 
          'from the true reward: {}, \n average per traj step: {}'.format(
           log_likelihood, log_likelihood / (n_traj * traj_len)))
    
    theta, losses, V, Q = max_causal_ent_irl(mdp, feature_matrix, trajectories, gamma, h, 
                               t_irl, epochs, learning_rate, q_opt = Q, v_opt = V, r = r_expert)
    print(Q)
    print(V)
    print('Final reward weights: ', theta)
    probs = torch.nn.functional.softmax( torch.matmul(torch.tensor(feature_matrix), theta),dim=1 )#.argmax(dim=1)
    print("Predicted Beliefs \n", (probs))
    
    plt.figure(dpi = 150)
    plt.plot(losses, label = "Predicted Reward")
    plt.xlabel("Optimization Iterations")
    plt.ylabel("Total Log Likelihood")
    plt.legend()
    plt.title("Log Likelihood of Trajectories over Time")
    plt.savefig("maxcent.png")

if __name__ == "__main__":
    main(traj_len=2, n_traj= 10000, epochs = 1000, learning_rate= .1, t_expert= 1, t_irl=1)