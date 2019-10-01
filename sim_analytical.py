import numpy as np
import matplotlib.pyplot as plt
import brian2 as br
import matplotlib.animation as animation
from matplotlib import rc
import scipy.special as scisp 
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


spike_size = 0.4
poisson_rate = 150
R = 10
C = 2
tau_val = R * C
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


def bernoulli_poisson(run_time,poisson_rate,threshold_val,tau_val,mu,eqs,spike_size,no_neurons, graph):

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
	pass_con = br.Synapses(P, input_neuron, 'w : 1', on_pre='v_post += '+str(spike_size)+'*sign((randn()+'+str(mu)+'))')
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
	spikes_of_neuron = br.SpikeMonitor(first)
	second_mon = br.StateMonitor(second, 'v', record=True)

	br.run(run_time*br.ms)

	if graph:

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


	pos_list = []
	neg_list = []
	for i in spikes_of_neuron.t:
		if first_mon.v[0,int(float(i)*10000)] > 0.1:
			pos_list.append(i)
		else:
			neg_list.append(i)
	#print(pos_list)
	#print(neg_list)
	time_delta_pos = np.diff(np.append([0], list(pos_list)))
	time_delta_neg = np.diff(np.append([0], list(neg_list)))
	return [np.average(time_delta_pos), np.std(time_delta_pos), np.average(time_delta_neg), np.std(time_delta_neg)]

bernoulli_poisson(800,150,1,R*C,0,eqs,0.4,1, True)


#plot poisson firing rate




#plot analict function
mu = 0
sigma = 2.0

sim_size = 40000
step_size = 0.02

#poisson
experimental_neg = []
analytical_neg = []
experimental_pos = []
analytical_pos = []
for i in np.arange(1.4,3.0,step_size):
	analytical_pos.append(scisp.erfinv((threshold_val * 2*C)/(np.pi*i))+0)
	temp = bernoulli_poisson(sim_size,i*100,1,R*C,0,eqs,0.3,1, False)
	experimental_pos.append(temp[0])
	analytical_neg.append(scisp.erfinv((threshold_val * 2*C)/(np.pi*i))+0)
	experimental_neg.append(temp[2])
plt.clf()
plt.plot(np.arange(1.4,3.0,step_size),analytical_pos,'--')
plt.plot(np.arange(140,300,step_size*100)/100,experimental_pos)
plt.savefig('figures/poisson_var_pos.png', bbox_inches='tight')
plt.clf()
plt.plot(np.arange(1.4,3.0,step_size),analytical_neg,'--')
plt.plot(np.arange(140,300,step_size*100)/100,experimental_neg)
plt.savefig('figures/poisson_var_neg.png', bbox_inches='tight')




#bernoulli
analytical_pos = []
experimental_pos = []
analytical_neg = []
experimental_neg = []
for i in np.arange(-0.8,0.8,step_size):
	temp = bernoulli_poisson(sim_size,1.6*100,1,R*C,i,eqs,0.3,1, False)
	analytical_pos.append(scisp.erfinv((threshold_val * 2*C)/(np.pi*1.6)-i))
	experimental_pos.append(temp[0])
	analytical_neg.append(-scisp.erfinv((-threshold_val * 2*C)/(np.pi*1.6)-i))
	experimental_neg.append(temp[2])
plt.clf()
plt.plot(np.arange(-0.8,0.8,step_size),analytical_pos,'--')
plt.plot(np.arange(-0.8,0.8,step_size),experimental_pos)
plt.savefig('figures/bernoulli_drift_pos.png', bbox_inches='tight')
plt.clf()
plt.plot(np.arange(-0.8,0.8,step_size),analytical_neg,'--')
plt.plot(np.arange(-0.8,0.8,step_size),experimental_neg)
plt.savefig('figures/bernoulli_drift_neg.png', bbox_inches='tight')








