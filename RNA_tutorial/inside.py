import numpy as np
from abc import ABC, abstractmethod
mfe_s = {"GC":-3.0, "CG":-3.0, "AU":-2.0, "UA":-2.0, "GU":-1.0, "UG":-1.0}
R = 0.0019872041
temperature = 298.15


class Semiring(float, ABC):
    zero = None
    one = None
    def check_semiring(self, other):
        if type(self) is not type(other):
            raise TypeError("cannot combine elements from different semirings!!")

    @abstractmethod
    def __add__(self, other):
        pass

    def __iadd__(self, other):
        return self+other

    @abstractmethod
    def __mul__(self, other):
        pass

    @classmethod
    @abstractmethod
    def energy_to_semiring_element(semiring_class, energy):
        pass

    @abstractmethod
    def priority(self):
        pass

class MFESemiring(Semiring):
    def __add__(self, other):
        self.check_semiring(other)
        return type(self)(min(float(self), float(other)))

    def __mul__(self, other):
        self.check_semiring(other)
        return type(self)(float(self)+float(other))

    @classmethod
    def energy_to_semiring_element(semiring_class, energy):
        return semiring_class(energy)

    def priority(self):
        return -float(self)

class PartitionSemiring(Semiring):
    def __add__(self, other):
        self.check_semiring(other)
        return type(self)(float(self)+float(other))

    def __mul__(self, other):
        self.check_semiring(other)
        return type(self)(float(self)*float(other))

    @classmethod
    def energy_to_semiring_element(semiring_class, energy):
        weight = np.exp(-energy/(R*temperature))
        return semiring_class(weight)

    def priority(self):
        return float(self)

class LogPartitionSemiring(Semiring):
    def __add__(self, other):
        self.check_semiring(other)
        return type(self)(np.logaddexp(float(self), float(other)))

    def __mul__(self, other):
        self.check_semiring(other)
        return type(self)(float(self)+float(other))

    @classmethod
    def energy_to_semiring_element(semiring_class, energy):
        weight = -energy/(R*temperature)
        return semiring_class(weight)

    def priority(self):
        return float(self)


def initialize_semiring_units():
    MFESemiring.zero = MFESemiring(float("inf"))
    MFESemiring.one = MFESemiring(0.0)
    PartitionSemiring.zero = PartitionSemiring(0.0)
    PartitionSemiring.one = PartitionSemiring(1.0)
    LogPartitionSemiring.zero = LogPartitionSemiring(float("-inf"))
    LogPartitionSemiring.one = LogPartitionSemiring(0.0)
initialize_semiring_units()


def valid_pair(sequence, i, j):
    return sequence[i] + sequence[j] in mfe_s

def update(states, key, value, semiring):
    if key not in states:
        states[key] = semiring.zero
    states[key] += value

def beam_prune(states, endpoints, beam_size, semiring):
    if beam_size == None:
        return states
    if len(states) <= beam_size:
        return states
    ranked = []
    for key, value in states.items():
        left_energy = endpoints[key].get(0, semiring.zero)
        total = left_energy*value
        state_priotity = total.priority()
        ranked.append((state_priotity, key, value))
    ranked.sort(key = lambda item: item[0], reverse=True)
    kept = ranked[:beam_size]
    return {i : value for _, i, value in kept}

def left_to_right(sequence, beam_size=100, semiring=None):
    if semiring == None:
        semiring = LogPartitionSemiring
    n = len(sequence)
    #would use default dict here, but might makes things messy by inserting keys
    endpoints = [{} for _ in range(n+1)]
    endpoints[0][0] = semiring.one
    for j in range(n):
        states = list(endpoints[j].items())
        for i, span_value in states:
            unpaired_candidate = span_value
            update(endpoints[j+1], i, unpaired_candidate, semiring)
            if i > 0 and valid_pair(sequence, i-1, j):
                parter_index = i-1
                left_states = list(endpoints[parter_index].items())
                for k, left_value in left_states:
                    energy = mfe_s[sequence[parter_index]+sequence[j]]
                    weight = semiring.energy_to_semiring_element(energy)
                    paired_candidate = span_value*left_value*weight
                    update(endpoints[j+1], k, paired_candidate, semiring)
        endpoints[j+1] = beam_prune(endpoints[j+1], endpoints, beam_size, semiring)
        endpoints[j+1][j+1] = semiring.one
    final_value = endpoints[n].get(0, semiring.zero)
    return final_value, endpoints
