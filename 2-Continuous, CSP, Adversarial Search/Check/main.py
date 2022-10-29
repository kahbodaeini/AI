import question2 as q2

tests = q2.get_all_tests(prefix='q2_')

n, m, m_next_lines, e, next_e_lines = q2.scan_test_input(tests[0])

preferred_halls_list = m_next_lines
constraints = next_e_lines

suitable_departments_for_each_hall = {i+1: [] for i in range(n)}
binary_constraints = {i+1: [] for i in range(n)}

for i in range(m):
    for j in preferred_halls_list[i]:
        suitable_departments_for_each_hall[j].append(i+1)

print(suitable_departments_for_each_hall)

# print(preferred_halls_list)
# print(constraints)

for i in range(e):
    binary_constraints[constraints[i][0]].append(constraints[i][1])

print(binary_constraints)


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
        lis = []
        for j in suitable_departments_for_each_hall[y]:
            lis.append([i, j])
        if sub_set(lis, constraints):
            suitable_departments_for_each_hall[x].remove(i)
            removed = True

    return removed


def ac_3(suitable_departments_for_each_hall, binary_constraints):
    #################################################################
    # (Point: 30% of total score obtained by tests)                 #
    # This function returns false                                   #
    # if an inconsistency is found and true otherwise.              #
    # Feel free to also implement a `revise` function in this cell. #
    #################################################################
    queue = constraints

    while len(queue) > 0:
        x, y = queue.pop(0)

        if remove_inconsistent_values(x, y):
            x_neighbors = binary_constraints[x]

            for i in x_neighbors:
                queue.append([i, x])

    return suitable_departments_for_each_hall, binary_constraints


kir, kir1 = ac_3(suitable_departments_for_each_hall, binary_constraints)
print("result")
print(kir)
print(kir1)