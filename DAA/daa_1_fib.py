#Experiment 1.Write a program to calculate Fibonacci numbers and find its step count. 
import timeit

# Iterative (non-recursive) Fibonacci
def fibonacci(n):
    fib_list = [0] * (n + 1)
    fib_list[0] = 0
    fib_list[1] = 1
    for i in range(2, n + 1):
        fib_list[i] = fib_list[i - 1] + fib_list[i - 2]
    return fib_list[n]

# Recursive Fibonacci with memoization
fib_recur_list = [0] * 100  # big enough array

def fibonacci_recursive(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if fib_recur_list[n] != 0:
        return fib_recur_list[n]
    fib_recur_list[n] = fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
    return fib_recur_list[n]

# Testing
N = int(input("Enter the value of N (e.g., 10 or 20): "))
RUNS = 1000
print(f"Given N = {N}\n{RUNS} runs")

# Non-recursive
print(
    "Fibonacci non-recursive:",
    fibonacci(N),
    "\tTime:",
    f'{timeit.timeit("fibonacci(N)", setup=f"from __main__ import fibonacci;N={N}", number=RUNS):.5f}s',
    "\tO(n)\tspace: O(1)"
)

# Recursive
print(
    "Fibonacci recursive:\t",
    fibonacci_recursive(N),
    "\tTime:",
    f'{timeit.timeit("fibonacci_recursive(N)", setup=f"from __main__ import fibonacci_recursive;N={N}", number=RUNS):.5f}s',
    "\tO(2^n)\tspace: O(n)"
)
