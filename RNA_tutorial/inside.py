import numpy as np
mfe_s = {"GC":-3.0, "CG":-3.0, "AU":-2.0, "UA":-2.0, "GU":-1.0, "UG":-1.0}
R = 0.0019872041
temperature = 298.15


class MFESemiring:
    zero = float("inf")
    one = 0.0

    def plus(self, value_1, value_2):
        return min(value_1, value_2)

    def times(self, value_1, value_2):
        return value_1 + value_2

    def unpaired(self, value):
        return value

    def paired(self, sequence, i, j, value_1, value_2):
        pair = sequence[i] + sequence[j]
        energy = mfe_s[pair]
        return self.times(self.times(energy, value_1), value_2)

    def priority(self, value):
        return -value


class PartitionSemiring:
    zero = 0.0
    one = 1.0

    def plus(self, value_1, value_2):
        return value_1 + value_2

    def times(self, value_1, value_2):
        return value_1 * value_2

    def unpaired(self, value):
        return value

    def paired(self, sequence, i, j, value_1, value_2):
        pair = sequence[i] + sequence[j]
        energy = mfe_s[pair]
        partition = np.exp(-energy/(R*temperature))
        return self.times(self.times(partition, value_1), value_2)

    def priority(self, value):
        return value



class LogPartitionSemiring:
    zero = float("-inf")
    one = 0.0

    def plus(self, value_1, value_2):
        return np.logaddexp(value_1, value_2)

    def times(self, value_1, value_2):
        return value_1 + value_2

    def unpaired(self, value):
        return value

    def paired(self, sequence, i, j, value_1, value_2):
        pair = sequence[i] + sequence[j]
        energy = mfe_s[pair]
        partition = -energy/(R*temperature)
        return self.times(self.times(partition, value_1), value_2)

    def priority(self, value):
        return value


def valid_pair(sequence, i, j):
    return sequence[i] + sequence[j] in mfe_s

def update(states, key, value, semiring):
    old = states.get(key, semiring.zero)
    states[key] = semiring.plus(value, old)

def beam_prune(states, endpoints, beam_size, semiring):
    if beam_size == None:
        return states
    if len(states) <= beam_size:
        return states
    ranked = []
    for key, value in states.items():
        left_energy = endpoints[key].get(0, semiring.zero)
        total = semiring.times(left_energy, value)
        state_priotity = semiring.priority(total)
        ranked.append((state_priotity, key, value))
    ranked.sort(key = lambda item: item[0], reverse=True)
    kept = ranked[:beam_size]
    return {i : value for _, i, value in kept}

def left_to_right(sequence, beam_size=100, semiring=None):
    if semiring == None:
        semiring = LogPartitionSemiring()
    n = len(sequence)
    endpoints = [{} for _ in range(n+1)]
    endpoints[0][0] = semiring.one
    for j in range(n):
        states = list(endpoints[j].items())
        for i, span_value in states:
            unpaired_candidate = semiring.unpaired(span_value)
            update(endpoints[j+1], i, unpaired_candidate, semiring)
            if i > 0 and valid_pair(sequence, i-1, j):
                parter_index = i-1
                left_states = list(endpoints[parter_index].items())
                for k, left_value in left_states:
                    paired_candidate = semiring.paired(sequence, parter_index, j, span_value, left_value)
                    update(endpoints[j+1], k, paired_candidate, semiring)
        endpoints[j+1] = beam_prune(endpoints[j+1], endpoints, beam_size, semiring)
        endpoints[j+1][j+1] = semiring.one
    final_value = endpoints[n].get(0, semiring.zero)
    return final_value, endpoints
