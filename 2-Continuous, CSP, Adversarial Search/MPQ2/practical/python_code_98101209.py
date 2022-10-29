student_number = 98101209
Name = 'Kahbod'
Last_Name = 'Aeini'

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

def f_1(x):
    return (pow(x, 2) * np.cos(x/10) - x)/100

def f_2(x):
    return np.log(np.sin(x/20))

def f_3(x):
    return np.log(np.cos(x) - 45/x)

def draw(func, x_range):
    xpoints = x_range
    ypoints = [func(i) for i in x_range]
    plt.plot(xpoints, ypoints)
    plt.xlabel("x")
    plt.ylabel("f1(x)")
    plt.show()

def gradiant_descent(func, initial_point: float, learning_rate: float, max_iterations: int):
    cur_x = initial_point
    iters = 0

    while iters < max_iterations:
        x = [cur_x - 0.1, cur_x - 0.05, cur_x, cur_x + 0.05, cur_x + 0.1]
        f = [func(i) for i in x]
        cur_x = cur_x - learning_rate * np.gradient(f)[2]
        iters = iters+1
    return func(cur_x)

def f(x_1, x_2):
    return 2*pow(x_1, 2) + 3*pow(x_2, 2) - 4*x_1*x_2 - 50*x_1 + 6*x_2

def gradiant_descent(func, initial_point: Tuple, learning_rate: float, threshold: float, max_iterations: int):
    x_1_sequence = [initial_point[0]]
    x_2_sequence = [initial_point[1]]
    
    
    iters = 0

    while iters < max_iterations:
        cur_x1 = x_1_sequence[-1]
        cur_x2 = x_2_sequence[-1]


        x1 = [cur_x1 - 0.1, cur_x1 - 0.05, cur_x1, cur_x1 + 0.05, cur_x1 + 0.1]
        x2 = [cur_x2 - 0.1, cur_x2 - 0.05, cur_x2, cur_x2 + 0.05, cur_x2 + 0.1]

        f = [func(x1[i], x2[i]) for i in range(5)]

        cur_x1 = cur_x1 - learning_rate * np.gradient(f)[2]
        cur_x2 = cur_x1 - learning_rate * np.gradient(f)[2]

        if cur_x1 < threshold and cur_x2 < threshold:
            x_1_sequence.append(cur_x1)
            x_2_sequence.append(cur_x2)
            iters = iters+1

        else:
            break
    return x_1_sequence, x_2_sequence
    

def update_points(func, x_1, x_2, learning_rate):
    pass

x1, x2 = gradiant_descent(func=f, initial_point=(-100, 100), learning_rate=0.01, threshold=100, max_iterations=1000)

draw_points_sequence(f, x_1_sequence=x1, x_2_sequence=x2)

x1, x2 = gradiant_descent(func=f, initial_point=(-100, 100), learning_rate=0.01, threshold=100, max_iterations=1000)

draw_points_sequence(f, x_1_sequence=x1, x_2_sequence=x2)

x1, x2 = gradiant_descent(func=f, initial_point=(-100, 100), learning_rate=0.19, threshold=100, max_iterations=1000)

draw_points_sequence(f, x_1_sequence=x1, x_2_sequence=x2)

x1, x2 = gradiant_descent(func=f, initial_point=(-100, 100), learning_rate=0.4, threshold=100, max_iterations=1000)

draw_points_sequence(f, x_1_sequence=x1, x_2_sequence=x2)


def sub_set(lis1, lis2):
    is_sub_set = True

    for i in lis1:
        if i in lis2:
            continue
        else:
            is_sub_set = False
            break

    return is_sub_set


def construct_neighbors(x):
    lis = []

    for i in binary_constraints:

        if i.key() == x:
            lis.append(i[1])

    return lis


def remove_inconsistent_values(x, y):
    removed = False
    
    for i in suitable_departments_for_each_hall[x]:
        if len(suitable_departments_for_each_hall[y]) == 1 and suitable_departments_for_each_hall[y] == i:
            suitable_departments_for_each_hall[x].remove(i)
            removed = True

    return removed

def ac_3():
    queue = constraints

    while len(queue) > 0:
        x, y = queue.pop(0)

        if remove_inconsistent_values(x, y):
            if not suitable_departments_for_each_hall[x]:
                return False
            for z in binary_constraints[x]:
                if z != y:
                    queue.append((x, z))

    return suitable_departments_for_each_hall, binary_constraints

def backtrack(suitable_departments_for_each_hall, binary_constraints, assignments):
    if len(assignments) == n:
        if len(assignments) == n:
            return assignments
        return
    
    for i in range(1, n + 1):
        if i not in assignments.keys():
            variable = i
    
    for value in suitable_departments_for_each_hall[variable]:
        assignments[variable] = value
    
    
    
        if ac_3():
            result = backtrack(suitable_departments_for_each_hall, binary_constraints, assignments)
            if result:
                return result
        del assignments[variable]
    return

def backtracking_search(suitable_departments_for_each_hall, binary_constraints, assignments):
    assignments = backtrack(suitable_departments_for_each_hall, binary_constraints, assignments)
    assignments_str = ''
    if not assignments:
        assignments_str = 'NO'
    else:
        assignments_ordered = []
        for i in range(1, n + 1):
            assignments_ordered.append(assignments[i])
        assignments_str = ' '.join(map(str, assignments_ordered))
    return assignments_str



class MinimaxPlayer(Player):
    
    def __init__(self, col, x, y):
        super().__init__(col, x, y)

    def checkCol(self, x):
        pass
    

    def checkRow(self, y):
        pass
    

    def moveU(self, x, y, board):
        pass
    

    def moveD(self, x, y, board):
        pass

    def moveR(self, x, y, board):
        pass
    
    def moveL(self, x, y, board):
        pass

    def moveUR(self, x, y, board):
        pass
    
    def moveUL(self, x, y, board):
        pass

    def moveDR(self, x, y, board):
        pass

    def moveDL(self, x, y, board):
        pass

    def canMove(self, x, y, board):
        pass

    def minValue(self, board, alpha, beta, depth):
        pass
    
    def maxValue(self, board, alpha, beta, depth):
        pass
    
    def getMove(self, board):
        alpha = float('-inf')
        beta = float('inf')
        next = IntPair(-20, -20)

        if (board.getNumberOfMoves() == board.maxNumberOfMoves):
            return IntPair(-20, -20)
        
        if not (self.canMove(board.getPlayerX(self.getCol()), board.getPlayerY(self.getCol()), board)):
            return IntPair(-10, -10)
        
        if (self.getCol() == 1):
            pass

        else:
            pass
        pass

p1 = NaivePlayer(1, 0, 0)
p2 = NaivePlayer(2, 7, 7)
g = Game(p1, p2)
numberOfMatches = 1
score1, score2 = g.start(numberOfMatches)
print(score1/numberOfMatches)









