# Experiment 4.  0-1 Knapsack Problem using Dynamic Programming

def knapsack(W, wt, val, n):
    # Create a 2D DP table
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom-up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # Return the maximum value that can be put in a knapsack of capacity W
    return K[n][W]


# Driver Code
val = [60, 100, 120]   # values (profits)
wt = [10, 20, 30]      # weights
W = 50                 # maximum capacity
n = len(val)

print("Maximum value that can be put in a knapsack =", knapsack(W, wt, val, n))
