# 8 Puzzle State Space Representation

# Initial and goal State
initial_state = [[1, 2, 3],
                 [4, 0, 6],
                 [7, 5, 8]]

goal_state = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

# Function to find blank position
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# Function to print puzzle
def print_state(state):
    for row in state:
        print(row)
    print()

# Move functions
def move_up(state):
    i, j = find_blank(state)
    if i > 0:
        state[i][j], state[i-1][j] = state[i-1][j], state[i][j]

def move_down(state):
    i, j = find_blank(state)
    if i < 2:
        state[i][j], state[i+1][j] = state[i+1][j], state[i][j]

def move_left(state):
    i, j = find_blank(state)
    if j > 0:
        state[i][j], state[i][j-1] = state[i][j-1], state[i][j]

def move_right(state):
    i, j = find_blank(state)
    if j < 2:
        state[i][j], state[i][j+1] = state[i][j+1], state[i][j]

print("Initial State:")
print_state(initial_state)

# Example moves
move_down(initial_state)
move_right(initial_state)

print("State after moves:")
print_state(initial_state)

# Goal Test
if initial_state == goal_state:
    print("Goal State Reached")
else:
    print("Goal State Not Reached")
