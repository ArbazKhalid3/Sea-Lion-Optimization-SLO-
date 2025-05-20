import numpy as np # type: ignore

# Objective Function (you can change it)
def sphere_function(position):
    return np.sum(position ** 2)

# Initialize Sea Lions
def initialize_sea_lions(pop_size, dim, lb, ub):
    return np.random.uniform(lb, ub, (pop_size, dim))

# Main SLO Algorithm
def sea_lion_optimization(obj_func, dim, lb, ub, pop_size=30, max_iter=100):
    # Initialize sea lions and their scores
    population = initialize_sea_lions(pop_size, dim, lb, ub)
    fitness = np.apply_along_axis(obj_func, 1, population)
    
    best_idx = np.argmin(fitness)
    best_position = population[best_idx].copy()
    best_score = fitness[best_idx]
    
    for t in range(max_iter):
        a = 2 * (1 - t / max_iter)  # Decreasing coefficient
        
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            
            if np.random.rand() < 0.5:
                D = np.abs(C * best_position - population[i])
                population[i] = best_position - A * D
            else:
                rand_index = np.random.randint(pop_size)
                rand_position = population[rand_index]
                D = np.abs(C * rand_position - population[i])
                population[i] = rand_position - A * D

            # Boundary check
            population[i] = np.clip(population[i], lb, ub)
        
        # Evaluate fitness
        fitness = np.apply_along_axis(obj_func, 1, population)
        current_best_idx = np.argmin(fitness)
        current_best_score = fitness[current_best_idx]

        if current_best_score < best_score:
            best_score = current_best_score
            best_position = population[current_best_idx].copy()

        print(f"Iteration {t+1}/{max_iter}, Best Score: {best_score:.5f}")
    
    return best_position, best_score

# Example use
if __name__ == "__main__":
    dim = 30
    lb = -10
    ub = 10
    best_pos, best_val = sea_lion_optimization(sphere_function, dim, lb, ub)
    print("Best Position:", best_pos)
    print("Best Fitness Value:", best_val)
