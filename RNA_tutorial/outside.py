import numpy as np
from inside import LogPartitionSemiring, left_to_right, update, valid_pair


def right_to_left(sequence, beam_size=100, semiring=None):
    if semiring == None:
        semiring = LogPartitionSemiring()
    _, inside = left_to_right(sequence, beam_size, semiring)
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
    return final_value, outside, inside


def base_pair_probs(sequence, beam_size=100):
    semiring = LogPartitionSemiring()
    _, outside, inside = right_to_left(sequence, beam_size, semiring)
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
                    log_probability = (paired_candiate + outside_parent)-inside[n].get(0, semiring.zero)
                    probability = np.exp(log_probability)
                    if (i-1, j) in probs:
                        probs[i-1, j] += probability
                    else:
                        probs[i-1, j] = probability
    return probs
