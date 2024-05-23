import math
import random

def ackley_function(x, y):
    a = 20
    b = 0.2
    c = 2 * math.pi
    part_one = -a * math.exp(-b * math.sqrt(0.5 * (x ** 2 + y ** 2)))
    part_two = -math.exp(0.5 * (math.cos(c * x) + math.cos(c * y))) + a + math.exp(1)
    ackley = part_one + part_two
    return ackley


def grid_search(num_iterations, step_size, x_range, y_range):
    best_x = random.uniform(-x_range, x_range)
    best_y = random.uniform(-y_range, y_range)
    best_solution = ackley_function(best_x, best_y)
    
    for i in range(x_range * 2):
        if i == 0:
            i += 1
        for j in range(y_range * 2):
            if j == 0:
                j += 1
            hill_x, hill_y, hill_sol = hill_climbing(num_iterations, step_size, i, j)
            if best_solution > hill_sol:
                best_solution = hill_sol
                best_x = hill_x
                best_y = hill_y
    return best_x, best_y, best_solution


def hill_climbing(num_iterations, step_size, x, y):
    best_x = random.uniform(-x, x)
    best_y = random.uniform(-y, y)
    best_solution = ackley_function(x, y)
    
    for i in range(num_iterations):
        new_x = best_x + random.uniform(-step_size, step_size)
        new_y = best_y + random.uniform(-step_size, step_size)
        new_solution = ackley_function(new_x, new_y)
        if best_solution > new_solution:
            best_x, best_y = new_x, new_y
            best_solution = new_solution

    return best_x, best_y, best_solution

x_range = int(input("Enter half of the perimeter length: ")) // 2
y_range = int(input("Enter half of the width length: ")) // 2
num_iterations = int(input("Please enter the number of iterations: "))
step_size = float(input("Please determine the step size: "))
best_x, best_y, best_solution = grid_search(num_iterations, step_size, x_range, y_range)
print("Best solution at (X =", best_x + x_range, " Y =", best_y + y_range, ") with value:", best_solution)
# Ackyly
# random number baraye x,y
#acky(x,y)
#best
#tekrar 
#x+step size random
#y + step size
#ackley(newx,y)
#best <> bozorg tar ya kochek tar
#best