from numpy import exp
import numpy as np
from scipy.optimize import newton
from matplotlib import pyplot as plt

#we assume a = 1 is always true
a, b, B = 1, 6, 4.5

# Equation 15
def z(lamb):
    return 1. / sum(exp(-k*lamb) for k in range(a, b + 1))

# Equation 16
def f(lamb, B=B):
    y = sum(k * exp(-k*lamb) for k in range(a, b + 1))
    return y * z(lamb) - B

# Equation 17
def p(k, lamb):
    return z(lamb) * exp(-k * lamb)

#This function returns a distribution over total roll given:
#the distribution over individual rolls and total number of rounds
#theres probably a cleaner way to do this with pascal's triangle or something :)
def score_dist(dist, rounds):
    #the "first" roll is always 0
    prior_sums = np.zeros(1)
    prior_probs = np.ones(1)
    
    for round in range(1,rounds+1):
        sums = np.arange(round, 1+(b*round), 1)
        probs = np.zeros(len(sums))
        
        #print(sums, probs)
        
        for roll in range(1, 1+b):
            for index in range(len(prior_sums)):
                #get prior sums and probabilitie
                sum = prior_sums[index]
                prob = prior_probs[index]
                
                #compute new sum given roll
                new_sum = sum + roll
                
                #compute probability of this sum given "roll"
                new_prob = prob * dist[roll-1]
                sum_index = int(new_sum-round)
                sums[sum_index] = new_sum
                probs[sum_index] += new_prob
        prior_probs = probs
        prior_sums = sums
    return sums, probs
                
        
#computes the expected return of dist 1 given dist 2
def expected_return(dist1, dist2, rounds):
    _, probs1 = score_dist(dist1, rounds)
    _, probs2 = score_dist(dist2, rounds)
    
    score = 0
    for i in range(len(probs1)):
        for j in range(len(probs2)):
            if i > j:
                score += probs1[i]*probs2[j]
            elif i < j:
                score -= probs1[i]*probs2[j]
    return score
                

if __name__ == "__main__":
    unbiased = [1/b for i in range(b)]
    rounds = 2
        
    print("Sanity check using unbiased, 6-sided die and 2 rolls")
    sums, probs = score_dist(unbiased, 2)
    print("Possible scores:", sums)
    print("Distribution over scores:", probs)
    print("-"*10)
    
    print("Sanity check that expected return works given two players with unbiased dice")
    print("Expected return:", expected_return(unbiased, unbiased, 2))
    print("Note the value may not be exactly 0 due to floating point error")
    print("-"*10)

    print("Now we are going to generate biased dice at various expected values")
    evs = np.arange(0, 2.5, .1)+3.5
    
    dists = []
    scores = np.zeros(len(evs))
    index = 0
    for ev in evs:
        lamb = newton(lambda val: f(val, ev) , x0=0.5)
        probs = np.zeros(b)
        for k in range(a, b + 1):
            probs[k-1] = p(k, lamb)
        #print(f"Probabilities given individual expexted value of {ev}: {probs}")
        score = expected_return(probs, unbiased, rounds)
        scores[index] = score
        index += 1
        dists.append(tuple(probs))
        #print(f"Expected return given {rounds} rounds of play: {score}" )
        #print()
    
    dists = np.array(dists).T
    plt.figure(dpi = 150)
    #xs = [ev - 3.5 for ev in evs]
    plt.plot(evs, scores, marker = "o")
    plt.title(f"Return given Different Biased Dice and {rounds} Rounds")
    plt.xlabel("Bias")
    plt.ylabel("Expected Return")
    #plt.xscale("log")
    plt.savefig("plot1.png")
    plt.close()
    
    plt.figure(dpi = 150)
    #xs = [ev - 3.5 for ev in evs]
    for i in range(len(dists)):
        plt.plot(evs, dists[i], label = str(i+1))
    plt.title(f"Maximum Entrpy Biased Die at Different Biases")
    plt.xlabel("Bias")
    plt.ylabel("Probability of Rolling")
    plt.legend()
    #plt.xscale("log")
    plt.savefig("plot1b.png")
    plt.close()
    print("-"*10)
    
    print("Now lets see what happens given a fixed advantage and a varying number of rounds")
    rounds = [i for i in range(1, 11)]
    scores = np.zeros(len(rounds))
    ev = 3.55
    lamb = newton(lambda val: f(val, ev) , x0=0.5)
    probs = np.zeros(b)
    for k in range(a, b + 1):
        probs[k-1] = p(k, lamb)
    index = 0
    for round in rounds:
        score = expected_return(probs, unbiased, round)
        scores[index] = score
        index += 1
        #print(f"Expected return given {round} rounds of play: {score}" )
        #print()
    
    plt.figure(dpi = 150)
    plt.plot(rounds, scores, marker = "o")
    plt.title(f"Return given Varying Rounds and a Bias of {ev}")
    plt.xlabel("# Rounds")
    plt.ylabel("Expected Return")
    #plt.xscale("log")
    plt.savefig("plot2.png")
    plt.close()
    
    print("Lets see the joint relationship between the number of rounds and the advantage")
    
    
    fig = plt.figure(dpi = 150)
    ax = fig.add_subplot(111, projection='3d')
    
    evs = [3.5 + (.05*i) for i in range(11)]
    rounds = [i*2 for i in range(1, 12)]
    scores = np.zeros(shape = (len(evs), len(rounds)))
    index1 = 0
    for ev in evs:
        index2 = 0
        for round in rounds:
            lamb = newton(lambda val: f(val, ev) , x0=0.5)
            probs = np.zeros(b)
            for k in range(a, b + 1):
                probs[k-1] = p(k, lamb)
            score = expected_return(probs, unbiased, round)
            scores[index1, index2] = score
            index2 += 1
        index1 += 1

    X, Y = np.meshgrid(rounds, evs)
    Z = scores.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_title("Expected Return for Different Biases and # Rounds")
    ax.set_xlabel('# Rounds')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Expected Return')
    plt.savefig("plot3.png")
    
    print("Lets find the optimal bias given our desired win rate")
    
    desired_return = .1
    threshold = 0.00001
    rounds = 2
    
    low = 3.5
    high = 6.0
   
    total_return = 0
    while True:
        mid = ((high+low)/2)
        lamb = newton(lambda val: f(val, mid) , x0=0.5)
        probs = np.zeros(b)
        for k in range(a, b + 1):
            probs[k-1] = p(k, lamb)
        #print(f"Probabilities given individual expexted value of {ev}: {probs}")
        score = expected_return(probs, unbiased, rounds)
        print(score, mid, end = "\r")
        if np.abs(score - desired_return) <= threshold:
            break
        elif score < desired_return:
            low = mid
        else:
            high = mid
    print(f"To achieve an average return of {desired_return} given {rounds} rounds we need a bias of {mid} and a biased die with probabilities {probs}")
    
    print("Finally let's see how optimal bias varies with desired return")
    
    desired_returns = np.arange(0, 1, .05)
    threshold = 0.00001
    rounds = 2
    biases = []
    
    for desired_return in desired_returns:
        low = 3.5
        high = 6.0
    
        total_return = 0
        while True:
            mid = ((high+low)/2)
            lamb = newton(lambda val: f(val, mid) , x0=0.5)
            probs = np.zeros(b)
            for k in range(a, b + 1):
                probs[k-1] = p(k, lamb)
            #print(f"Probabilities given individual expexted value of {ev}: {probs}")
            score = expected_return(probs, unbiased, rounds)
            print(score, mid, end = "\r")
            if np.abs(score - desired_return) <= threshold:
                break
            elif score < desired_return:
                low = mid
            else:
                high = mid
        biases.append(mid)
        
    plt.figure(dpi = 150)
    plt.plot(desired_returns, biases, marker = "o")
    plt.xlabel("Desired Expected Return")
    plt.ylabel("Necessary Bias")
    plt.title("Die Bias in terms of Desired Return")
    plt.savefig("plot4.png")
    plt.close()