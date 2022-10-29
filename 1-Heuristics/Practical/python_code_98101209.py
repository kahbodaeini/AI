student_number = 98101209
Name = 'Kahbod'
Last_Name = 'Aeini'



def solve(N, M, K, NUMS, roads): 
    pass

def manhattan_dis(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def absolute_dis(point1, point2):
    return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

def heur_displaced(state):    
    displaced_boxes = 0
    storages = list(state.storage.keys())
    for boxes in state.boxes.keys():
        if boxes not in storages:
            displaced_boxes += 1
    return displaced_boxes

def heur_manhattan_distance(state):
    total = 0
    for box_coordinates, box_index in state.boxes.items():
        
        all_dis = {storages : manhattan_dis(box_coordinates, storages) for storages in state.storage}
        
        sorted_distances = sorted(all_dis.items(), key=lambda x: x[1])
        if state.restrictions is None:
            total = total + sorted_distances[0][1]
        else:
            for storage, distance in sorted_distances:
                if storage in state.restrictions[box_index]:
                    total = total + distance
                    break
    return total

def heur_euclidean_distance(state):  
    total = 0
    for box_coordinates, box_index in state.boxes.items():
        
        all_dis = {storages : absolute_dis(box_coordinates, storages) for storages in state.storage}
        
        sorted_distances = sorted(all_dis.items(), key=lambda x: x[1])
        if state.restrictions is None:
            total = total + sorted_distances[0][1]
        else:
            for storage, distance in sorted_distances:
                if storage in state.restrictions[box_index]:
                    total = total + distance
                    break
    return total



import time
def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    best_path_cost = float("inf")
    time_remain = 8
    optimal_final = None
    first_time = True

    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se = SearchEngine('custom', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)

    while (time_remain > 0) and not se.open.empty():
        start_time = time.time()
        final = se.search(timebound, (best_path_cost, best_path_cost, best_path_cost))
        if (not optimal_final and final) or (final and final.gval <= optimal_final.gval):
            best_path_cost = final.gval
            optimal_final = final
        if first_time:
            first_time = False
        time_remain -= (time.time() - start_time)
    try:
        return optimal_final
    except:
        return final





edge_count = np.sum(graph_matrix)/2 # Complete This (1 Points)
print(edge_count)

def random_state_generator(n):
    lis = []
    for i in range(n):
        lis.append(random.randint(0, 1))
    return lis

def neighbour_state_generator(state):
    new_state = state.copy()
    length = len(state)
    vertex_to_change = random.randrange(0, length - 1)
    previous_value = state[vertex_to_change]
    new_state[vertex_to_change] = 1 - state[vertex_to_change]
    return new_state, previous_value, vertex_to_change

def cost_function(graph_matrix,state , A = 1 , B=1):
    cost = A * sum(_ for _ in state)
    for i in range(len(graph_matrix)):
        for j in range(len(graph_matrix)):
            cost =  cost + (B * graph_matrix[i][j] * (1 - (state[i] or state[j])))
    return cost

deg = [np.sum(i) / edge_count for i in graph_matrix] #Complete This (2 Points)

def prob_accept(current_state_cost, next_state_cost, T, changed_index, previous_value): 
    if next_state_cost <= current_state_cost:
        return 1
    delta_f = next_state_cost - current_state_cost
    if previous_value == 1:
        return np.exp(-((delta_f * (1 - deg[changed_index])) / T))
    return np.exp(-((delta_f * (1 + deg[changed_index])) / T))

def accept(current_state , next_state , T, changed_index, previous_value):
    current_state_cost = cost_function(graph_matrix, current_state)
    next_state_cost = cost_function(graph_matrix, next_state)
    p = prob_accept(current_state_cost , next_state_cost , T, changed_index, previous_value)
    return False if random.random() > p else True

def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.show()

plot_cost(cost_list)

def random_state_generator2(n):
    lis = []
    for i in range(n):
        lis.append(random.randint(0, 1))
    return lis

def population_generation(n, k): 
    return [random_state_generator2(n) for _ in range(k)]

def cost_function2(graph,state):
    return cost_function(graph, state, B=2.5)

def tournament_selection(graph, population):
    new_population = []
    for i in range(len(population) // 2):
        if cost_function2(graph, population[i]) > cost_function2(graph, population[len(population) - i - 1]):
            new_population.append(population[i])
        else:
            new_population.append(population[len(population) - i - 1])
    return new_population

def crossover(graph, parent1, parent2):
    index = random.randint(0, len(graph) - 1)
    return parent1[:(index + 1)] + parent2[(index + 1):], parent2[:(index + 1)] + parent1[(index + 1):]

def mutation(chromosme,probability):
    mutated_chromosome = chromosme.copy()
    if random.random() < probability:
        index = random.randint(0, len(chromosme) - 1)
        mutated_chromosome[index] = 1 - mutated_chromosome[index]
    return mutated_chromosome

def genetic_algorithm(graph_matrix,mutation_probability=0.1,pop_size=100,max_generation=100):
    best_cost = None
    best_solution = None
    population = population_generation(len(graph_matrix), pop_size)
    for i in range(max_generation):
        selected_population = tournament_selection(graph_matrix , population)
        new_population = selected_population.copy()
        for j in range(len(selected_population) // 2):
            child1, child2 = crossover(graph_matrix, selected_population[j], selected_population[len(selected_population) - j - 1])
            new_population.extend([mutation(child1, mutation_probability), mutation(child2, mutation_probability)])
        for state in new_population:
            if (not best_cost) or cost_function2(graph_matrix, state) < best_cost:
                best_cost = cost_function2(graph_matrix, state) 
                best_solution = state
        population = new_population
    return best_cost, best_solution

