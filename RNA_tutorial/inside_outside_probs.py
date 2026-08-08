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

def left_to_right(sequence, beam_size, semiring):
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

mfe, end_points_mfe = left_to_right( sequence="CCCGGG",beam_size=100,semiring=MFESemiring())
partition, end_points_par = left_to_right( sequence="CCCGGG",beam_size=100,semiring=PartitionSemiring())
log_partition, end_points_logpar = left_to_right( sequence="CCCGGG",beam_size=100,semiring=LogPartitionSemiring())


def right_to_left(sequence, inside, semiring):
    n = len(sequence)
    outside = [{} for _ in range(n+1)]
    outside[n][0] = semiring.one
    for j in range(n-1, -1, -1):
        states = list(inside[j].items())
        for i, inside_energy in states:
            unpaired_candidate = outside[j+1].get(i, semiring.zero)
            if unpaired_candidate != semiring.zero:
                update(outside[j], i, unpaired_candidate, semiring)
            if i > 0 and valid_pair(sequence, i-1, j):
                partner_index = i-1
                left_states = list(inside[partner_index].items())
                for k, left_value in left_states:
                    outside_parent = outside[j+1].get(k, semiring.zero)
                    if outside_parent == semiring.zero:
                        continue
                    inside_child_outside = semiring.paired(sequence, partner_index, j, outside_parent, left_value)
                    left_child_outside = semiring.paired(sequence, partner_index, j, outside_parent, inside_energy)
                    update(outside[j], i, inside_child_outside, semiring)
                    update(outside[partner_index], k, left_child_outside, semiring)
    final_value = outside[0].get(0, semiring.zero)
    return final_value, outside


mfe_outside, outside_mfe = right_to_left( sequence="CCCGGG", inside=end_points_mfe, semiring=MFESemiring())
partition_outside, outside_par = right_to_left( sequence="CCCGGG", inside=end_points_par, semiring=PartitionSemiring())
log_partition_outside, outside_logpar = right_to_left( sequence="CCCGGG", inside=end_points_logpar, semiring=LogPartitionSemiring())

#better to change this to use log values by changing line 167 to
#probaibility = (paired_candiate + outside_parent)-inside[n].get(0, semiring.zero)
#then finally make sure you use the log results as arguments
def base_pair_probs(sequence, outside, inside, semiring):
    n = len(sequence)
    probs = {}
    for j in range(n-1, -1, -1):
        states = list(inside[j].items())
        for i, inside_value in states:
            if i > 0 and valid_pair(sequence, i-1, j):
                partner_index = i-1
                left_states = list(inside[partner_index].items())
                for k, left_value in left_states:
                    outside_parent = outside[j+1].get(k, semiring.zero)
                    if outside_parent == semiring.zero:
                        continue
                    paired_candiate = semiring.paired(sequence, partner_index, j, left_value, inside_value)
                    probability = (paired_candiate * outside_parent)/inside[n].get(0, semiring.zero)
                    if (i-1, j) in probs:
                        probs[i-1, j] += probability
                    else:
                        probs[i-1, j] = probability
    return probs

probs = base_pair_probs( sequence="CCCGGG", inside=end_points_par, semiring=PartitionSemiring(), outside=outside_par)
