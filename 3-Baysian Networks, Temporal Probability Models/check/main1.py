import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)

given_cpts = {'a': {1: 0.8, 0: 0.2},
              'b': {1: 0.55, 0: 0.45},
              'e': {1: {('b', 1): 0.3, ('b', 0): 0.9}, 0: {('b', 1): 0.7, ('b', 0): 0.1}},
              'd': {1: {(('a', 0), ('c', 0)): 0.8, (('a', 0), ('c', 1)): 0.65
                    , (('a', 1), ('c', 0)): 0.5, (('a', 1), ('c', 1)): 0.67},
                    0: {(('a', 0), ('c', 0)): 0.2, (('a', 0), ('c', 1)): 0.35
                        , (('a', 1), ('c', 0)): 0.5, (('a', 1), ('c', 1)): 0.33}},
              'c': {1: {(('a', 0), ('e', 0)): 0.7, (('a', 0), ('e', 1)): 0.15
                    , (('a', 1), ('e', 0)): 0.5, (('a', 1), ('e', 1)): 0.05},
                    0: {(('a', 0), ('e', 0)): 0.3, (('a', 0), ('e', 1)): 0.85
                    , (('a', 1), ('e', 0)): 0.5, (('a', 1), ('e', 1)): 0.95}},
              'f': {1: {('d', 1): 0.2, ('d', 0): 0.25}, 0: {('d', 1): 0.8, ('d', 0): 0.75}}}

prob = []

t = given_cpts

for a in range(2):
    for b in range(2):
        for c in range(2):
            for d in range(2):
                for e in range(2):
                    for f in range(2):
                        prob.append(t['a'][a] * t['b'][b] * t['e'][e][('b', b)] * t['d'][d][(('a', a), ('c', c))]
                                    * t['c'][c][(('a', a), ('e', e))] * t['f'][f][('d', d)])


d = {'A': [0 if i % 64 < 32 else 1 for i in range(64)],
     'B': [0 if i % 32 < 16 else 1 for i in range(64)],
     'C': [0 if i % 16 < 8 else 1 for i in range(64)],
     'D': [0 if i % 8 < 4 else 1 for i in range(64)],
     'E': [0 if i % 4 < 2 else 1 for i in range(64)],
     'F': [0 if i % 2 < 1 else 1 for i in range(64)],
     "prob": prob}
table = pd.DataFrame(data=d)
print(table)

for index, row in table.iterrows():
    print(list(row))

print(set([('b', 0), ('c', 1)]))