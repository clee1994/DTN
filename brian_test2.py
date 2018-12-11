import numpy as np
import matplotlib.pyplot as plt
import brian2 as br
import matplotlib.animation as animation
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


spike_size = 0.4
poisson_rate = 150
tau_val = 20
threshold_val = 1



run_time = 800
no_neurons = 1


eqs = '''
dv/dt = (-v)/tau : 1
tau : second
'''


########
#negative constant current
########

eqsI = '''
dv/dt = (I-v)/tau : 1
tau : second
I:1
'''

br.start_scope()


first = br.NeuronGroup(no_neurons, eqsI, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')
second = br.NeuronGroup(no_neurons, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')

first.tau = [tau_val]*br.ms
second.tau = [tau_val]*br.ms
first.I = [-1.001]

S2 = br.Synapses(first, second, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*'+str(spike_size))
S2.connect(j='i')


first_mon = br.StateMonitor(first, 'v', record=True)
spikes_of_neuron = br.SpikeMonitor(first)
second_mon = br.StateMonitor(second, 'v', record=True)

br.run(run_time*br.ms)




plt.clf()
plt.subplot(3, 1, 1)
#plt.plot(first_mon.t/br.ms, first_mon.v[0], label='Input Current')
plt.hlines(y=-1.001,xmin=0,xmax=run_time,color='b',linestyles="-")
#plt.plot()
plt.ylabel('Input Current')
#plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
#plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.subplot(3, 1, 2)
plt.plot(first_mon.t/br.ms, first_mon.v[0], label='1st Neuron')
plt.ylabel('1st Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.subplot(3, 1, 3)
plt.plot(second_mon.t/br.ms, second_mon.v[0], label='2st Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('2nd Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.savefig('figures/static.png', bbox_inches='tight')

########
#positive poisson input
########
br.start_scope()

P = br.PoissonGroup(no_neurons, poisson_rate*br.Hz)

first = br.NeuronGroup(no_neurons, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')
second = br.NeuronGroup(no_neurons, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')


first.tau = [tau_val]*br.ms
second.tau = [tau_val]*br.ms


S1 = br.Synapses(P, first, 'w : 1', on_pre='v_post += '+str(spike_size))
S2 = br.Synapses(first, second, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*'+str(spike_size))
S1.connect(j='i')
S2.connect(j='i')

first_mon = br.StateMonitor(first, 'v', record=True)
spikes_of_neuron = br.SpikeMonitor(P)
second_mon = br.StateMonitor(second, 'v', record=True)

br.run(run_time*br.ms)

spike_times = [float(i) for i in list(spikes_of_neuron.t)]

plt.clf()
plt.subplot(3, 1, 1)
#plt.plot(first_mon.t/br.ms, first_mon.v[0], label='Spikes')
markerline, stemlines, baseline = plt.stem(np.multiply(spike_times,1000), [spike_size]*len(spike_times),markerfmt=",")
plt.setp(baseline, 'color', 'b')
plt.ylabel('Input Spike')
#plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
#plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.subplot(3, 1, 2)
plt.plot(first_mon.t/br.ms, first_mon.v[0], label='1st Neuron')
plt.ylabel('1st Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.subplot(3, 1, 3)
plt.plot(second_mon.t/br.ms, second_mon.v[0], label='2st Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('2nd Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.savefig('figures/poisson.png', bbox_inches='tight')


########
#bernoulli poisson input
########


br.start_scope()



P = br.PoissonGroup(no_neurons, poisson_rate*br.Hz)
input_neuron = br.NeuronGroup(no_neurons, eqs, threshold='v>0 or v<-0', reset='v = 0', method='euler')
first = br.NeuronGroup(no_neurons, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')
second = br.NeuronGroup(no_neurons, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')


input_neuron.tau = [tau_val]*br.ms
first.tau = [tau_val]*br.ms
second.tau = [tau_val]*br.ms

# Comment these two lines out to see what happens without Synapses
# dirak delta function -> exponential decay


#find out characteristics -> poisson both ways two dimensional random variable
# -> two dimensional two one dimensional

 # analyuze what this is
pass_con = br.Synapses(P, input_neuron, 'w : 1', on_pre='v_post += '+str(spike_size)+'*sign((randn()))')
S1 = br.Synapses(input_neuron, first, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*'+str(spike_size))
S2 = br.Synapses(first, second, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*'+str(spike_size))

#'''
#v_post += (v/abs(v))*0.2 : 1
#v = 0 : 1
#''')
pass_con.connect(j='i')
S1.connect(j='i')
S2.connect(j='i')
#S.w = '0.3'

in_mon = br.StateMonitor(pass_con, 'v', record=True)
first_mon = br.StateMonitor(first, 'v', record=True)
second_mon = br.StateMonitor(second, 'v', record=True)

br.run(run_time*br.ms)


plt.clf()
plt.subplot(3, 1, 1)

plt.plot(in_mon.t/br.ms, in_mon.v[0], label='Input')
#plt.xlabel('Time (ms)')
plt.ylabel('Input Spike')

plt.subplot(3, 1, 2)
plt.plot(first_mon.t/br.ms, first_mon.v[0], label='1st Neuron')

#plt.xlabel('Time (ms)')
plt.ylabel('1st Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.subplot(3, 1, 3)
plt.plot(second_mon.t/br.ms, second_mon.v[0], label='2st Neuron')
plt.xlabel('Time (ms)')
plt.ylabel('2nd Neuron')
plt.hlines(y=1,xmin=0,xmax=run_time,color='r',linestyles='dashed')
plt.hlines(y=-1,xmin=0,xmax=run_time,color='r',linestyles='dashed')

plt.savefig('figures/double.png', bbox_inches='tight')


#time_delta = np.diff(np.append([0], list(spikes_of_neuron.t)))
#print(np.average(time_delta))
#print(np.std(time_delta))


