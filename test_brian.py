from brian2 import *





start_scope()
tau = 10*ms

eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs)
run(100*ms)

start_scope()

G = NeuronGroup(1, eqs, method='exact')
M = StateMonitor(G, 'v', record=True)

run(30*ms)

plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')
show()


start_scope()

N= 1
tau = 10*ms
v0_max = 3.
eqs = '''
dv/dt = (v0-v)/tau : 1
v0 : 1
'''

G = NeuronGroup(N, eqs, threshold='v>0.8 or v<-0.8', reset='v = 0', method='euler')
M = StateMonitor(G, 'v', record=0)
G.v0 = '5'


run(50*ms)


clf()
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v')
savefig("test.png")





start_scope()

N = 1000
tau = 10*ms
vr = -70*mV
vt0 = -50*mV
delta_vt0 = 5*mV
tau_t = 100*ms
sigma = 0.5*(vt0-vr)
v_drive = 2*(vt0-vr)
duration = 100*ms

eqs = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0-vt)/tau_t : volt
'''

reset = '''
v = vr
vt += delta_vt0
'''

G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset, refractory=5*ms, method='euler')
spikemon = SpikeMonitor(G)

G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0

run(duration)

_ = hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=ones(len(spikemon))/(N*defaultclock.dt))
xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)');
savefig("test.png")



start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

G = NeuronGroup(2, eqs, threshold='v>1 or v<-1', reset='v = 0', method='exact')
G.I = [2, 0]
G.tau = [10, 100]*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

M = StateMonitor(G, 'v', record=True)

run(100*ms)

plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
xlabel('Time (ms)')
ylabel('v')
legend()







start_scope()

indices = array([0, 1, 2])
times = array([1, 2, 3])*ms
G = SpikeGeneratorGroup(1, indices, times)

S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)







br.start_scope()

stimulus = br.TimedArray([[0.1,0.2,0.05,0.3],[10,20,10,30]],dt=1*br.ms)

eqs = '''
dv/dt = (-v)/tau : 1
I : 1
tau : second
'''


G = br.NeuronGroup(2, 'dv/dt = (-v + stimulus(t))/(10*ms) : 1', threshold='v>1', reset='v = 0', method='exact')
#G.I = [2, 0]
#G.tau = [10, 100]*br.ms

# Comment these two lines out to see what happens without Synapses
S = br.Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

M = br.StateMonitor(G, 'v', record=True)

br.run(100*br.ms)

plt.plot(M.t/br.ms, M.v[0], label='Neuron 0')
plt.plot(M.t/br.ms, M.v[1], label='Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('v')




#legend();

import brian2 as br

ta = br.TimedArray([1, 2, 3, 4] * br.mV, dt=0.1*br.ms)
#print(ta(0.3*br.ms))
G = br.NeuronGroup(1, 'v = ta(t) : volt')
mon = br.StateMonitor(G, 'v', record=True)
net = br.Network(G, mon)
net.run(1*br.ms)  # doctest: +ELLIPSIS
print(mon[0].v)


ta2d = TimedArray([[1, 2], [3, 4], [5, 6]]*mV, dt=0.1*ms)
G = NeuronGroup(4, 'v = ta2d(t, i%2) : volt')
mon = StateMonitor(G, 'v', record=True)
net = Network(G, mon)
net.run(0.2*ms)  # doctest: +ELLIPSIS
print(mon.v[:])

