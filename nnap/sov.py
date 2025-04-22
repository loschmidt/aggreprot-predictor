import click
import numpy as np
import pandas as pd


def find_sequences(arr) -> list:
    groups = []

    start = 0
    stop = 0

    for i in range(1, len(arr)):
        x = arr[i]
        z = arr[i - 1]
        if x - z != 1:
            stop = i
            groups.append((start, stop))
            start = i
            if i + 1 == len(arr):
                stop = i + 1
                groups.append((start, stop))
            stop = 0
        elif x - z == 1 and i + 1 == len(arr):
            stop = i + 1
            groups.append((start, stop))

    sequences = [arr[i[0]:i[1]] for i in groups]
    return sequences


def delta(*args):
    args = np.array(args)
    return args.min()


def find_pairs(SGt, SP) -> tuple:
    output = []
    for i in SGt:
        counter = 0
        for z in SP:
            for y in z:
                if y in i:
                    counter += 1
            if counter > 0:
                output.append((i, z))
                counter = 0

    return output


def find_not_connected_sequences_length(pairs, SGt) -> int:
    output = []
    pairsSGT = list(zip(*pairs))[0]

    counter = 0
    for i in SGt:
        for z in pairsSGT:
            if np.array_equal(i, z):
                counter += 1
        if counter == 0:
            output.append(len(i))
        counter = 0

    return sum(output)


def max_OV(itter: np.ndarray, not_itter: np.ndarray) -> int:
    itter_first = itter[0]
    itter_last = itter[len(itter) - 1]
    not_itter_last = not_itter[len(not_itter) - 1]
    if itter_last < not_itter_last:
        return (not_itter_last - itter_first) + 1
    elif itter_last >= not_itter_last:
        return (itter_last - itter_first) + 1


def SOV_measure_i(i: str, SGt: np.ndarray, SP: np.ndarray) -> (int, np.ndarray, np.ndarray):
    SGti = np.where(SGt == i)[0]
    if len(SGti) <= 0:
        return (1, [0], [0])

    SPi = np.where(SP == i)[0]
    if len(SPi) <= 0:
        return (1, [0], [0])

    sequence_SGti = find_sequences(SGti)
    sequence_SPi = find_sequences(SPi)
    pairs = find_pairs(sequence_SGti, sequence_SPi)
    if len(pairs) > 0:
        non_connected_pairs_length = find_not_connected_sequences_length(pairs, sequence_SGti)
    else:
        return (sum([len(i) for i in sequence_SGti]), [0], [0])
    lensSGti = []
    fractions = []

    # minov
    for x, z in pairs:
        to_tteration, not_tterate = (x, z) if x[0] < z[0] else (z, x)
        minOv = 0
        for z_i in to_tteration:
            if z_i in not_tterate:
                minOv += 1

        # maxOv
        maxOv = max_OV(to_tteration, not_tterate)

        # lenS1
        lenSGti = len(x)
        lensSGti.append(lenSGti)
        # lenS2
        lenSPi = len(z)
        delta_i = delta((maxOv - minOv), minOv, (int(0.5 * lenSGti)), (int(0.5 * lenSPi)))
        fraction = (minOv + delta_i) / maxOv
        fraction *= lenSGti
        fractions.append(fraction)

    N_i = sum(lensSGti) + non_connected_pairs_length
    return N_i, fractions, lensSGti


@click.command()
def example():
    # searched value
    i_chain = 'C'
    # sequences
    SGt = ['C', 'C', 'C', 'H', 'H', 'H', 'C', 'C', 'H', 'H', 'C', 'C']
    SP = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']

    SGt = np.array(SGt)
    SP = np.array(SP)
    # sov measure
    N_i, fractions, lenSGti = SOV_measure_i(i_chain, SGt, SP)
    SOV_measure = sum(fractions) / N_i
    print(f"SOV measure for {i_chain} is {round(SOV_measure * 100, 3)}%")





if __name__ == "__main__":
    measure()
    # example()
