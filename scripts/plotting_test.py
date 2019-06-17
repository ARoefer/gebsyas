#!/usr/bin/env python
from random import random, normalvariate

from gebsyas.plotting import draw_recorders, split_recorders, ValueRecorder

if __name__ == '__main__':
    recorders = [ValueRecorder('Some data {}'.format(x), *['Group {}'.format(y) for y in range(int(random() * 5) + 1)]) for x in range(6)]

    for r in recorders:
        for d in r.data.keys():
            mean = normalvariate(0, 100)
            variance = abs(normalvariate(1, 6))
            r.data[d] = [normalvariate(mean, variance) for x in range(30)]
        r.compute_limits()

    draw_recorders(split_recorders(recorders, 0.05) + recorders, 1).savefig('plotting_test.png')