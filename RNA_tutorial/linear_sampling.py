import numpy as np
from inside import LogPartitionSemiring, left_to_right, valid_pair
from outside import base_pair_probs


def recover_edges(sequence, inside, semiring, i, end):
    edges = []
    j = end - 1
    skip = inside[j].get(i, semiring.zero)
    if skip != semiring.zero:
        edges.append(("unpaired", skip, (i, j)))
    for k in range(i, j):
        if valid_pair(sequence, k, j):
            left_weight = inside[k].get(i, semiring.zero)
            inside_weight = inside[j].get(k+1, semiring.zero)
            paired = semiring.paired(sequence, k, j, left_weight, inside_weight)
            if paired != semiring.zero:
                edges.append(("paired", paired, (k, j), (i, k), (k+1, j)))
    return edges


def pick_structure(sequence, inside, structure, edge_cache, semiring, probs, i, end, rng):
    if i >= end:
        return
    state = (i, end)
    if state not in edge_cache:
        edge_cache[state] = recover_edges(sequence, inside, semiring, i, end)
    edges = edge_cache[state]
    log_weights = np.array([edge[1] for edge in edges], dtype=float)
    max_log = np.max(log_weights)
    weights = np.exp(log_weights - max_log)
    probabilities = weights / weights.sum()
    chosen_edge = edges[rng.choice(len(edges), p=probabilities)]
    if chosen_edge[0] == "unpaired":
        child_i, child_end = chosen_edge[2]
        pick_structure(sequence, inside, structure, edge_cache, semiring, probs, child_i, child_end, rng)
    else:
        left, right = chosen_edge[2]
        structure[left] = "("
        structure[right] = ")"
        probs[left, right] = probs.get((left, right), 0) + 1
        left_i, left_end = chosen_edge[3]
        inside_i, inside_end = chosen_edge[4]
        pick_structure(sequence, inside, structure, edge_cache, semiring, probs, left_i, left_end, rng)
        pick_structure(sequence, inside, structure, edge_cache, semiring, probs, inside_i, inside_end, rng)

def linear_sampling(sequence, num_samples, beam_size):
    semiring = LogPartitionSemiring()
    _, inside = left_to_right(sequence, beam_size, semiring)
    edge_cache = {}
    samples = []
    probs = {}
    rng = np.random.default_rng()
    for _ in range(num_samples):
        structure = ["."] * len(sequence)
        pick_structure(sequence, inside, structure, edge_cache, semiring, probs, 0, len(sequence), rng)
        samples.append("".join(structure))
    for pair in probs:
        probs[pair] /= num_samples
    return samples, probs


def compare_probs(inside_outside_probs, sampled_probs):
    pairs = set(inside_outside_probs) | set(sampled_probs)
    total_error = 0
    for pair in pairs:
        expected = inside_outside_probs.get(pair, 0.0)
        sampled = sampled_probs.get(pair, 0.0)
        error = abs(expected - sampled)
        total_error += error
    average_error = total_error / len(pairs)
    return average_error

def samples_comparison(sequence, beam_size):
    probs = base_pair_probs(sequence, beam_size)
    for k in [1, 10, 100, 1000, 10000, 50000, 100000]:
        samples, sampled_probs = linear_sampling(sequence, k, beam_size)
        error = compare_probs(probs, sampled_probs)
        print("samples:", k, "error:", error)


if __name__ == "__main__":
    samples_comparison(sequence="AGGCAUCAAACCCUGCAUGGGAGCG", beam_size=100)
