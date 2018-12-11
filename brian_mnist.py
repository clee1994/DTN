import eventvision as ev
import numpy as np
import matplotlib.pyplot as plt
import brian2 as br
import matplotlib.animation as animation

spike_size = 0.2
tau_val = 1000
threshold_val = 1

eqs = '''
dv/dt = (-v)/tau : 1
tau : second
'''


#skim clasifier 83.44
test_file = "/Users/clemens/Dropbox/Documents/Deep Learning/Project/N-MNIST/Train/0/00022.bin" 


td = ev.read_dataset(test_file)

td.data = np.unique(td.data)

# # check whether there is False
# counter = 0
# for i in td.data:
# 	if i[2] == False:
# 		counter = counter +1 

flat_indices = (td.data.x * td.width) +  td.data.y

br.start_scope()


pos_spikes = br.SpikeGeneratorGroup(td.width * td.height, flat_indices[td.data.p == True], td.data.ts[td.data.p == True]*br.ms)
neg_spikes = br.SpikeGeneratorGroup(td.width * td.height, flat_indices[td.data.p == False], td.data.ts[td.data.p == False]*br.ms)




# input layer 
Input_Neurons = br.NeuronGroup(td.width * td.height, eqs, threshold='v>0 or v<-0', reset='v = 0', method='euler')
# hidden layer
Hidden_Neurons = br.NeuronGroup(50, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')
# output layer
Output_Neurons = br.NeuronGroup(10, eqs, threshold='v>'+str(threshold_val)+' or v<-'+str(threshold_val)+'', reset='v = 0', method='euler')

Input_Neurons.tau = [10]*br.ms
Hidden_Neurons.tau = [10]*br.ms
Output_Neurons.tau = [10]*br.ms


pos_input = br.Synapses(pos_spikes, Input_Neurons, 'w : 1', on_pre='v_post += w')
neg_input = br.Synapses(neg_spikes, Input_Neurons, 'w : 1', on_pre='v_post -= w')

S1 = br.Synapses(Input_Neurons, Hidden_Neurons, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*w')
S2 = br.Synapses(Hidden_Neurons, Output_Neurons, 'w : 1', on_pre='v_post += v_pre/abs(v_pre)*w')



pos_input.connect()
neg_input.connect()
S1.connect()
S2.connect()

#weights 
pos_input.w = str(spike_size)
neg_input.w = str(spike_size)
S1.w = str(spike_size)
S2.w = str(spike_size)

#P = br.PoissonGroup(1, 3000*br.Hz)
#first = br.NeuronGroup(1, eqs, threshold='v>1 or v<-1', reset='v = 0', method='euler')
#second = br.NeuronGroup(1, eqs, threshold='v>1 or v<-1', reset='v = 0', method='euler')



# Comment these two lines out to see what happens without Synapses
# dirak delta function -> exponential decay


#find out characteristics -> poisson both ways two dimensional random variable
# -> two dimensional two one dimensional
 # analyuze what this is

#'''
#v_post += (v/abs(v))*0.2 : 1
#v = 0 : 1
#''')

#S.w = '0.3'

#in_mon = br.StateMonitor(S1, 'v_post', record=True)
#first_mon = br.StateMonitor(first, 'v', record=True)
final_mon = br.StateMonitor(Output_Neurons, 'v', record=True)



br.run(max(td.data.ts)*br.ms)


plt.subplot(1, 1, 1)
plt.plot(final_mon.t[70000:120000]/br.ms, final_mon.v[1,70000:120000], label='Final Neuron')
plt.show()









# pulse_series = np.diff(np.append([0], in_mon.v_post[0]))
# pulse_series[pulse_series > 0.1] = 0
# pulse_series[pulse_series < -0.1] = 0
# pulse_series[pulse_series > 0.05] = 0.1
# pulse_series[pulse_series < -0.05] = -0.1
# #pulse_series[pulse_series < 0.05 and pulse_series > -0.05] = 0
# #pulse_series[pulse_series > -0.05] = 0


# plt.plot(in_mon.t/br.ms, pulse_series, label='Input')
# #plt.plot(Ms.t/br.ms, Ms.v[0], label='Neuron 1')
# plt.xlabel('Time (ms)')
# plt.ylabel('Spike')
# #plt.hlines(y=1,xmin=0,xmax=400,color='r',linestyles='dashed')
# #plt.hlines(y=-1,xmin=0,xmax=400,color='r',linestyles='dashed')

# plt.subplot(3, 1, 2)
# 
# #plt.plot(Ms.t/br.ms, Ms.v[0], label='Neuron 1')
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.hlines(y=1,xmin=0,xmax=400,color='r',linestyles='dashed')
# plt.hlines(y=-1,xmin=0,xmax=400,color='r',linestyles='dashed')

# plt.subplot(3, 1, 3)
# plt.plot(second_mon.t/br.ms, second_mon.v[0], label='2st Neuron')
# #plt.plot(Ms.t/br.ms, Ms.v[0], label='Neuron 1')
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.hlines(y=1,xmin=0,xmax=400,color='r',linestyles='dashed')
# plt.hlines(y=-1,xmin=0,xmax=400,color='r',linestyles='dashed')

# plt.show()