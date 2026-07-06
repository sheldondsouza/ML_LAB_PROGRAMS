# Hill Climbing Algorithm

def objective_function(x):
    return -(x - 5) ** 2 + 25


def hill_climbing(start):
    current = start

    while True:
        left = current - 1
        right = current + 1

        current_value = objective_function(current)
        left_value = objective_function(left)
        right_value = objective_function(right)

        # Choose the best neighbor
        if left_value > current_value:
            current = left
        elif right_value > current_value:
            current = right
        else:
            # Local optimum found
            return current, current_value


start_state = 0
best_state, best_value = hill_climbing(start_state)

print("Start State:", start_state)
print("Best State:", best_state)
print("Maximum Value:", best_value)