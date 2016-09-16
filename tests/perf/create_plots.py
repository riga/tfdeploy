# -*- coding: utf-8 -*-

"""
Plotting script that plots all kinds of distributions using the json file created by
measure_runtimes.py.
"""

import os
import json
import itertools
import collections

import matplotlib.pyplot as plt

from measure_runtimes import specs, output_file


#
# constants
#

plotdir = "data"


#
# plot function
#

def plot(xkey, coords, data):
    # determine the plotfile path
    labels = sorted([(xkey, 0)] + coords.items(), key=lambda tpl: specs.keys().index(tpl[0]))
    title  = ", ".join("%s: %s" % tpl for tpl in labels if tpl[0] != "examples" and tpl[0] != xkey)
    labels = [(key[0], value) for key, value in labels]
    plotfile = "runtime_" + xkey + "_"  + "_".join("%s%s" % tpl for tpl in labels) + ".png"
    plotfile = os.path.join(plotdir, plotfile)

    # get x and y data to plot
    x = [d[xkey] for d in data]
    ykeys = ["tf_cpu", "tf_gpu", "td"]
    y = {ykey: [d["times"][ykey]["mean"] for d in data] for ykey in ykeys}
    markers = dict(zip(ykeys, ("s", "o", "D")))

    # do the plot
    fig, axes = plt.subplots()
    for ykey, _y in y.items():
        axes.plot(x, _y, "-" + markers[ykey], label=ykey)

    axes.set_xlabel(xkey)
    axes.set_ylabel("time per batch [s]")
    axes.set_title(title)
    if xkey != "units":
        axes.set_xscale("log")
    axes.legend(loc="best")

    fig.savefig(plotfile)
    fig.clf()


#
# data filter function
#

def filter_data(data, **kwargs):
    return [d for d in data if all(d[key] == value for key, value in kwargs.items())]


#
# main and entry hook
#

def main():
    # check if the output file exists
    if not os.path.exists(output_file):
        IOError("output file '%s' does not exist, run run_tests.py to generate it" % output_file)

    # read the data
    with open(output_file, "r") as f:
        data = json.load(f)

    # prepare the plot dir
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    # loop through all keys in specs, each of them will be used for the x-axis
    for xkey in specs:
        # do not plot examples on the x-axis
        if xkey == "examples":
            continue

        # determine the remaining keys, do the combinatorics
        keys = [key for key in specs if key != xkey]
        for combi in itertools.product(*[specs[key] for key in keys]):
            coord = collections.OrderedDict(zip(keys, combi))
            plot_data = filter_data(data, **coord)
            plot_data.sort(key=lambda d: d[key])
            plot(xkey, coord, plot_data)

if __name__ == "__main__":
    main()
