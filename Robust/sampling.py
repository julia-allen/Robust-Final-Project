import numpy as np

def sample_ps(p):
    #for a given vector of probabilities, return a z-vector of 0's or 1's
        #sampled from uniform dist w prob p_i
    sample=np.random.binomial(10000,p)
    z=np.where(sample>=5000,1,0)
    return z

def sample_selected(x,p):
    #for a given set of selected stations (x_i=1 if selected) and probabilities p, return y=xz
        #where y=1 iff station i is selected and sampled to be feasible
    z=sample_ps(p)
    return x*z

def check_feas(p,alpha,x_star):
    #for a given x_star (stations opened) and p, get the new s-vector (prob of each neighborhood satisfied)
    # Initialize an empty list to store the probabilities for each column
    column_probabilities = []

    # Iterate over each column in alpha
    for col in alpha.T:
        # Filter the probabilities based on the entries of the column
        filtered_col=col*x_star
        filtered_p = p[filtered_col == 1]
        # Calculate the probability that at least one event is true in the filtered probabilities
        prob_at_least_one_true = 1 - np.prod(1 - filtered_p)
        # Append the result to the list of column probabilities
        column_probabilities.append(prob_at_least_one_true)

    #print("Probability of at least one event being true for each column:", column_probabilities)

    dot=np.dot(alpha.T,p)
    s=np.where(dot>0,1,0)
    return column_probabilities

if __name__ == "__main__":
    p=[0.9,0.5,0.9,0.9,0.5]
    x=[1,0,0,1,1]
    alpha=np.array([[0,1,0,1,0,1],[1,0,0,0,1,0],[0,1,1,0,0,0],[0,0,1,0,0,0],[0,1,1,0,1,0]])
    y=sample_selected(x,p)
    print(y)
    s=check_feas(y,alpha)
    #print(alpha)
    print(s)
