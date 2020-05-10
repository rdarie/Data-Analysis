# encoding: utf-8
# based on https://github.com/NeuralEnsemble/PyNN/blob/0.9.4/examples/small_network.py
"""
Small network created with the Population and Projection classes


Usage: small_network.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  plot the simulation results to a file
  --debug DEBUG  print debugging information

"""

import numpy
from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.parameters import Sequence
from pyNN.random import RandomDistribution as rnd

sim, options = get_simulator(
    ("--plot-figure", "Plot the simulation results to a file.", {"action": "store_true"}),
    ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

# === Define parameters ========================================================

# simulation
dt         = 0.1           # (ms)
simtime    = 1000.0        # (ms)

#  population
n = 20      # Number of cells

#  cellular biophysics
cell_params = {
    'tau_m'      : 20.0,   # (ms)
    'tau_syn_E'  : 2.0,    # (ms)
    'tau_syn_I'  : 4.0,    # (ms)
    'e_rev_E'    : 0.0,    # (mV)
    'e_rev_I'    : -70.0,  # (mV)
    'tau_refrac' : 2.0,    # (ms)
    'v_rest'     : -60.0,  # (mV)
    'v_reset'    : -70.0,  # (mV)
    'v_thresh'   : -50.0,  # (mV)
    'cm'         : 0.5}    # (nF)
#  synapses
syn_delay  = 1.0           # (ms)
w = 0.002  # synaptic weight (µS)

# input activity
input_rate = 50.0          # (Hz)

# === Setup the simulation ====================================================

sim.setup(timestep=dt, max_delay=syn_delay)
numpy.random.seed(26278342)

# === Define the cell types ====================================================

# In PyNN, the system of equations that defines a neuronal model is encapsulated in a CellType class.
neuron_type = sim.IF_cond_alpha(**cell_params)

# === Build the network ========================================================

'''
    http://neuralensemble.org/docs/PyNN/neurons.html#populations
    To create a Population, we need to specify at minimum the number of neurons and the cell type.
    Three additional arguments may optionally be specified:
        the spatial structure of the population;
        initial values for the neuron state variables;
        a label.
'''
cells = sim.Population(
    n, neuron_type,
    initial_values={'v': rnd('uniform', (-60.0, -50.0))},
    label="cells")

number = int(2 * simtime * input_rate / 1000.0)

def generate_spike_times(i):
    # The Sequence class represents a sequence of numerical values.
    gen = lambda: Sequence(numpy.add.accumulate(numpy.random.exponential(1000.0 / input_rate, size=number)))
    if hasattr(i, "__len__"):
        return [gen() for j in i]
    else:
        return gen()
assert generate_spike_times(0).max() > simtime

# Spike source generating spikes at the times given in the spike_times array.
spike_source = sim.Population(n, sim.SpikeSourceArray(spike_times=generate_spike_times))

'''
    http://neuralensemble.org/docs/PyNN/connections.html
    Analogously to neuron models, the system of equations that defines a synapse model
    is encapsulated in a SynapseType class.
    weights are in microsiemens or nanoamps,
        depending on whether the post-synaptic mechanism implements a change in conductance or current,
    and delays are in milliseconds
'''
syn = sim.StaticSynapse(weight=w, delay=syn_delay)
'''
    http://neuralensemble.org/docs/PyNN/connections.html#projections
    A Projection is a container for a set of connections between two populations of neurons
    Creating a Projection in PyNN also creates the connections at the level of the simulator.
    To create a Projection we must specify:
        he pre-synaptic population;
        he post-synaptic population;
         connection/wiring method;
         synapse type
    Optionally, we can also specify:
        the name of the post-synaptic mechanism (e.g. ‘excitatory’, ‘NMDA’) (by default, this is ‘excitatory’);
        a label (autogenerated if not specified);
        a Space object, which determines how distances should be calculated
            for distance-dependent wiring schemes or parameter values.
'''
input_conns = sim.Projection(spike_source, cells, sim.FixedProbabilityConnector(0.5), syn)

# === Select what to save ======================================================
spike_source.record('spikes')
cells.record('spikes')
cells[0:2].record(('v', 'gsyn_exc'))
# === Run simulation ===========================================================

sim.run(simtime)

filename = normalized_filename(
    "Results", "small_network", "pkl",
    options.simulator, sim.num_processes())
cells.write_data(filename, annotations={'script_name': __file__})

print("Mean firing rate: ", cells.mean_spike_count() * 1000.0 / simtime, "Hz")

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    figure_filename = filename.replace("pkl", "png")
    data = cells.get_data().segments[0]
    vm = data.filter(name="v")[0]
    gsyn = data.filter(name="gsyn_exc")[0]
    Figure(
        Panel(vm, ylabel="Membrane potential (mV)"),
        Panel(gsyn, ylabel="Synaptic conductance (uS)"),
        Panel(data.spiketrains, xlabel="Time (ms)", xticks=True),
        annotations="Simulated with %s" % options.simulator.upper()
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
