# 8-Queens Problem using Backtracking

N = 8  # size of chessboard (8x8)

def print_solution(board):
    for i in range(N):
        for j in range(N):
            print(board[i][j], end=" ")
        print()
    print()

# Function to check if a queen can be placed at board[row][col]
def is_safe(board, row, col):
    # Check this column on upper side
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row, -1, -1), range(col, N)):
        if board[i][j] == 1:
            return False

    return True

# Recursive function to solve the N-Queens problem
def solve_queens(board, row):
    # base case: if all queens are placed
    if row >= N:
        print("Final 8-Queens Matrix:")
        print_solution(board)
        return True

    for col in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1  # place queen
            if solve_queens(board, row + 1):
                return True  # if next queens placed, return success
            board[row][col] = 0  # backtrack

    return False

# Driver code
board = [[0 for _ in range(N)] for _ in range(N)]

# Place first queen manually at (0,0) (first row, first column)
board[0][0] = 1

# Start from row 1 because first queen is already placed
if not solve_queens(board, 1):
    print("Solution does not exist")
