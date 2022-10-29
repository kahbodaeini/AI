import pandas as pd
import random


class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class

    Attributes
    ----------
    n: int
        number of variables

    G: dict
        Network representation as a dictionary.
        {variable:[[children],[parents]]} # You can represent the network in other ways. This is only a suggestion.
    """

    def __init__(self) -> None:
        ############################################################
        # Initialzie Bayesian Network                              #
        # (1 Points)                                               #
        ############################################################

        # Your code
        self.n = 7
        self.G = {'a': [['d', 'c'], []], 'b': [['e'], []], 'c': [['d', 'g'], ['a', 'e']], 'd': [['g', 'f'], ['a', 'c']],
                  'e': [['c', 'g'], ['b']], 'f': [[], ['d']], 'g': [[], ['d', 'c', 'e']]}
        self.given_cpts = {'a': {1: 0.8, 0: 0.2},
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
        self.joint_cpt = {}
        t = self.given_cpts

        self.prob = []

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        for e in range(2):
                            for f in range(2):
                                self.prob.append(
                                    t['a'][a] * t['b'][b] * t['e'][e][('b', b)] * t['d'][d][(('a', a), ('c', c))]
                                    * t['c'][c][(('a', a), ('e', e))] * t['f'][f][('d', d)])

        d = {'A': [0 if i % 64 < 32 else 1 for i in range(64)],
             'B': [0 if i % 32 < 16 else 1 for i in range(64)],
             'C': [0 if i % 16 < 8 else 1 for i in range(64)],
             'D': [0 if i % 8 < 4 else 1 for i in range(64)],
             'E': [0 if i % 4 < 2 else 1 for i in range(64)],
             'F': [0 if i % 2 < 1 else 1 for i in range(64)],
             "prob": self.prob}
        self.table = pd.DataFrame(data=d)

    def cpt(self, node, value) -> dict:
        """
        This is a function that returns cpt of the given node

        Parameters
        ----------
        node:
            a variable in the bayes' net

        Returns
        -------
        result: dict
            {value1:{{parent1:p1_value1, parent2:p2_value1, ...}: prob1, ...}, value2: ...}
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################

        # Your code
        return self.given_cpts[node][value]

    def pmf(self, query, evidence) -> float:

        ############################################################
        # (3 Points)                                               #
        ############################################################

        # Your code
        query_probability = 0
        evidence_probability = 0

        for index, row in self.table.iterrows():

            query_set = set(query)
            evidence_set = set(evidence)

            # evidence_set = {}
            # for i in evidence:
            #     evidence_set.add(i)
            union = set.union(query_set, evidence_set)

            row_set = {('a', row['A']), ('b', row['B']), ('c', row['C']), ('d', row['D']), ('e', row['E']),
                       ('f', row['F']), ("prob", 'a', row["prob"])}

            if union.issubset(row_set):
                query_probability += row["prob"]

            if evidence_set.issubset(row_set):
                evidence_probability += row["prob"]

        return round(query_probability / evidence_probability, 6)

    def sampling(self, query, evidence, sampling_method, num_iter, num_burnin=1e2) -> float:

        ############################################################
        # (27 Points)                                              #
        #     Prior sampling (6 points)                            #
        #     Rejection sampling (6 points)                        #
        #     Likelihood weighting (7 points)                      #
        #     Gibbs sampling (8 points)                      #
        ############################################################

        # Your code

        if sampling_method == "Prior":

            ret = []

            for i in range(num_iter):
                a = random.choices([0, 1], weights=(0.2, 0.8))

                b = random.choices([0, 1], weights=(0.45, 0.55))

                e_prob = self.pmf([('e', 1)], [('b', b[0])])
                e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

                c_prob = self.pmf(('c', 1), [('e', e[0]), ('a', a[0])])
                c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

                d_prob = self.pmf(('d', 1), [('a', a[0]), ('c', c[0])])
                d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

                f_prob = self.pmf(('f', 1), [('d', d[0])])
                f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

                ret.append({('a', a[0]), ('b', b[0]), ('e', e[0]), ('c', c[0]), ('d', d[0]), ('f', f[0])})

            query_probability = 0
            evidence_probability = 0

            query_set = set(query)
            evidence_set = set(evidence)
            union = set.union(query_set, evidence_set)
            print(query_set)
            print(evidence_set)
            print(union)

            for i in ret:
                if union.issubset(i):
                    query_probability += 1

                if evidence_set.issubset(i):
                    evidence_probability += 1

            return round(query_probability / evidence_probability, 6)


        elif sampling_method == "Rejection":

            # a, b, e, c, d, f = -1

            # for ev in evidence:
            #     if ev[0] == 'a':
            #         a = ev[1]
            #         continue
            #     if ev[0] == 'b':
            #         b = ev[1]
            #         continue
            #     if ev[0] == 'e':
            #         e = ev[1]
            #         continue
            #     if ev[0] == 'c':
            #         c = ev[1]
            #         continue
            #     if ev[0] == 'd':
            #         d = ev[1]
            #         continue
            #     if ev[0] == 'f':
            #         f = ev[1]
            #         continue

            query_set = set(query)
            evidence_set = set(evidence)
            union = set.union(query_set, evidence_set)

            ret = []
            i = 0
            while True:

                if i == num_iter:
                    break

                # if a == -1:
                a = random.choices([0, 1], weights=(0.2, 0.8))

                # if b == -1:
                b = random.choices([0, 1], weights=(0.45, 0.55))

                # if e == -1:
                e_prob = self.pmf(('e', 1), [('b', b[0])])
                e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

                # if c == -1:
                c_prob = self.pmf(('c', 1), [('e', e[0]), ('a', a[0])])
                c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

                # if d == -1:
                d_prob = self.pmf(('d', 1), [('a', a[0]), ('c', c[0])])
                d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

                # if f == -1:
                f_prob = self.pmf(('f', 1), [('d', d[0])])
                f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

                if {('a', a[0]), ('b', b[0]), ('e', e[0]), ('c', c[0]), ('d', d[0]), ('f', f[0])}.issubset(evidence_set):
                    ret.append({('a', a[0]), ('b', b[0]), ('e', e[0]), ('c', c[0]), ('d', d[0]), ('f', f[0])})
                    i += 1

            query_probability = 0
            evidence_probability = 0

            for i in ret:
                if union.issubset(i):
                    query_probability += 1

                if evidence_set.issubset(i):
                    evidence_probability += 1

            return round(query_probability / evidence_probability, 6)

        elif sampling_method == "Likelihood Weighting":

            query_set = set(query)
            evidence_set = set(evidence)
            union = set.union(query_set, evidence_set)
            a = [0]
            b = [0]
            c = [0]
            d = [0]
            e = [0]
            f = [0]
            wi = 1
            w = []
            ret = []

            for ev in evidence:
                if ev[0] == 'a':
                    a = ev[1]
                    wi *= self.pmf(('a', a[0]), [])
                    continue
                if ev[0] == 'b':
                    b = ev[1]
                    wi *= self.pmf(('b', b[0]), [])
                    continue
                if ev[0] == 'e':
                    e = ev[1]
                    wi *= self.pmf(('e', e[0]), [('b', 1)])
                    continue
                if ev[0] == 'c':
                    c = ev[1]
                    wi *= self.pmf(('c', c[0]), [('e', 1), ('a', 1)])
                    continue
                if ev[0] == 'd':
                    d = ev[1]
                    wi *= self.pmf(('d', d[0]), [('a', 1), ('c', 1)])
                    continue
                if ev[0] == 'f':
                    f = ev[1]
                    wi *= self.pmf(('f', f[0]), [('d', 1)])
                    continue
            for i in range(num_iter):
                if a == -1:
                    a = random.choices([0, 1], weights=(0.2, 0.8))

                if b == -1:
                    b = random.choices([0, 1], weights=(0.45, 0.55))

                if e == -1:
                    e_prob = self.pmf(('e', 1), [('b', b[0])])
                    e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

                if c == -1:
                    c_prob = self.pmf(('c', 1), [('e', e[0]), ('a', a[0])])
                    c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

                if d == -1:
                    d_prob = self.pmf(('d', 1), [('a', a[0]), ('c', c[0])])
                    d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

                if f == -1:
                    f_prob = self.pmf(('f', 1), [('d', d[0])])
                    f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

                ret.append({('a', a[0]), ('b', b[0]), ('e', e[0]), ('c', c[0]), ('d', d[0]), ('f', f[0])})
                w.append(wi)

            query_probability = 0

            for i in range(len(ret)):
                if union.issubset(ret[i]):
                    query_probability += w[i]

            return round(query_probability / sum(w), 6)

        else:

            query_set = set(query)
            evidence_set = set(evidence)
            union = set.union(query_set, evidence_set)
            ret = []
            a = [0]
            b = [0]
            c = [0]
            d = [0]
            e = [0]
            f = [0]

            for ev in evidence:
                if ev[0] == 'a':
                    a[0] = ev[1]
                    continue
                if ev[0] == 'b':
                    b[0] = ev[1]
                    continue
                if ev[0] == 'e':
                    e[0] = ev[1]
                    continue
                if ev[0] == 'c':
                    c[0] = ev[1]
                    continue
                if ev[0] == 'd':
                    d[0] = ev[1]
                    continue
                if ev[0] == 'f':
                    f[0] = ev[1]
                    continue

            if a == -1:
                a = random.choices([0, 1], weights=(0.2, 0.8))

            if b == -1:
                b = random.choices([0, 1], weights=(0.45, 0.55))

            if e == -1:
                e_prob = self.pmf(('e', 1), [('b', b[0])])
                e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

            if c == -1:
                c_prob = self.pmf(('c', 1), [('e', e[0]), ('a', a[0])])
                c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

            if d == -1:
                d_prob = self.pmf(('d', 1), [('a', a[0]), ('c', c[0])])
                d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

            if f == -1:
                f_prob = self.pmf(('f', 1), [('d', d[0])])
                f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

            for i in range(num_burnin):

                stable = ['a', 'b', 'e', 'c', 'd', 'f'].pop

                if not stable == 'a':
                    a = random.choices([0, 1], weights=(0.2, 0.8))

                if not stable == 'b':
                    b = random.choices([0, 1], weights=(0.45, 0.55))

                if not stable == 'e':
                    e_prob = self.pmf(('e', 1), [('b', b)])
                    e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

                if not stable == 'c':
                    c_prob = self.pmf(('c', 1), [('e', e), ('a', a)])
                    c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

                if not stable == 'd':
                    d_prob = self.pmf(('d', 1), [('a', a), ('c', c)])
                    d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

                if not stable == 'f':
                    f_prob = self.pmf(('f', 1), [('d', d)])
                    f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

            for i in range(num_iter):

                stable = ['a', 'b', 'e', 'c', 'd', 'f'].pop

                if not stable == 'a':
                    a = random.choices([0, 1], weights=(0.2, 0.8))

                if not stable == 'b':
                    b = random.choices([0, 1], weights=(0.45, 0.55))

                if not stable == 'e':
                    e_prob = self.pmf(('e', 1), [('b', b)])
                    e = random.choices([0, 1], weights=(1 - e_prob, e_prob))

                if not stable == 'c':
                    c_prob = self.pmf(('c', 1), [('e', e), ('a', a)])
                    c = random.choices([0, 1], weights=(1 - c_prob, c_prob))

                if not stable == 'd':
                    d_prob = self.pmf(('d', 1), [('a', a), ('c', c)])
                    d = random.choices([0, 1], weights=(1 - d_prob, d_prob))

                if not stable == 'f':
                    f_prob = self.pmf(('f', 1), [('d', d)])
                    f = random.choices([0, 1], weights=(1 - f_prob, f_prob))

                ret.append({('a', a[0]), ('b', b[0]), ('e', e[0]), ('c', c[0]), ('d', d[0]), ('f', f[0])})

            query_probability = 0
            evidence_probability = 0

            for i in ret:
                if union.issubset(i):
                    query_probability += 1

                if evidence_set.issubset(i):
                    evidence_probability += 1

            return round(query_probability / evidence_probability, 6)


queries = [{('f', 1)}, {('c', 0), ('b', 1)}]
evidences = [{('a', 1), ('e', 0)}, {('f', 1), ('d', 0)}]
sampling_methods = ["Prior", "Rejection", "Likelihood Weighting", "Gibbs"]
itters = [100, 500, 1000, 3000, 10000, 50000]


# extra space
bn = BN()
for i in sampling_methods:
    for j in itters:
        print(i, j, "samples", bn.sampling([('f', 1)], [('a', 1), ('e', 0)], i, j, 10))
