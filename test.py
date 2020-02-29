import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve


kds = np.geomspace(0.01,1e8,1000) # in nM

def normalDistribution(avg,std):
    def cal(x):
        return 1/(std* (2.5066282746310002)) * np.exp(-0.5*((x-avg)/std)**2)
    return cal

n = normalDistribution(3,0.5)

f = n((np.log10(kds)))
f = f/f.sum()
sum(f)
conc = 1e15 * f
max(conc)

aptcKd = kds[conc/6.02e23/100e-6>1e-15]
aptc = conc[conc/6.02e23/100e-6>1e-15]/6e23/100e-6*1e9
len(aptc)
aptc.max()
aptc.sum()
def multiBinding(T,aptConc,aptKd,T0):
    return (T*aptConc/(T+aptKd)).sum() + T - T0

T0 = 10 # nM

res = fsolve(multiBinding,T0/2,args=(aptc,aptcKd,T0))

multiBinding(TFree,aptc,aptcKd,T0)
multiBinding(TFree,aptc,aptcKd,T0)

aptcKd.max()
aptcKd.min()
res
TFree
TFree = res[0]
aptFree = aptcKd * aptc / (TFree+ aptcKd)

aptCplx = aptFree * TFree / aptcKd
sum(aptCplx)+TFree

aptPercentBinding = 1- aptFree / aptc
aptPercentBinding

plt.plot(np.log10(aptcKd),aptPercentBinding*100)

calculatedPercentBinding = TFree/(TFree+aptcKd)

(calculatedPercentBinding - aptPercentBinding).max()

a = np.array([[1,2],[3,4]])
a.sum(axis=1)
a[:,0]
a= np.array([0.1,0.01,0,100,2,3])
(a>1).sum()
(a>1).nonzero()
a.nonzero()
b = a[a<3]
b
a[a<3][b<1]=np.array([1,2,3])
a
np.abs(a-10).argmin()
a.apply(lambda x: x)

np.apply_along_axis(a,)
plt.plot(np.log10(kds),f)
fig,ax = plt.subplots()
ax.plot(kds,conc)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Kd / nM')
ax.set_ylabel('Count in Pool')
ax.set_title('a')

np.random.randint(2)
np.random.random()
conc[0]/6e23/100e-6
bool(1e-19-1)
int()
(np.pi*2)**0.5

def binding(y,t,kf,kr):
    """
    binding A + T = AT;
    y in the order of A, T, AT
    """
    a ,t, at = y
    dy_dt = [
        - kf*a*t + kr*at,
        - kf*a*t + kr*at,
        kf*a*t - kr*at
    ]
    return dy_dt

def binding_ode(kf,kd,start_conc,period):
    """
    provide Kf : nM-1*S-1; 1e-4
    kd: nM
    start_conc:[ A/nM, T/nM, AT/nM]
    period: minutes
    """
    kr = kf * kd  # s-1
    thalf = np.log(2)/kr
    print("Kd {}nM".format(kd))
    print("Dissociate half life {:.2f} hrs".format(thalf/3600))
    print("Starting with {}nM A, {}nM T, {}nM AT".format(*start_conc))
    print('Simulate duration {} minutes'.format(period))
    print("Binding half life by kobs {:.2f} minutes".format(np.log(2)/60/(kf*start_conc[0])))
    timepoints = np.linspace(1,period*60,1000)
    result=odeint(binding,start_conc,timepoints,args=(kf,kr))
    fig,ax = plt.subplots()
    ax.plot(timepoints/60,[i[0] for i in result],)
    ax.set_xlabel('Time / min')
    ax.set_ylabel('Conc. / nM')
    ax.legend(['A','T','AT'])
    ax.grid()
    plt.tight_layout()
    plt.show()

binding_ode(kf=1e-2,kd=1,start_conc=[1.66e-8,1,0],period=5)

r = np.random.rand(10000000)

np.zeros_like(a).astype(float)

a = np.array([1,2,0,1,0,3])
a.dtype
(a/12).dtype
a[a.nonzero()]
a.nonzero()
n = np.empty_like(a)
n.fill(0.01)
n
a[a.nonzero()]

e = np.log(1/r)

_=plt.hist(r,histtype='step',bins=100,density=1)
_=plt.hist(e,histtype='step',bins=1000,density=1)


# an event have probability of p happen in unit time T; the time it take for it to happen
# next time is exponential distribution, with Ω = p * e^(-pt) (the time it take an event to happen
# in a poisson process is exponential distribution)

def happenTimeCompare(p,n = 100000,xlim=1000):
    """
    compare exponential distribution and simulation.
    """
    def happenTimeSimu(p):
        """simulate 10000 times"""
        def howlongtaketohappen(p):
            n=1
            while np.random.rand()>p:
                n+=1
            return n
        time = [howlongtaketohappen(p) for i in range(n)]
        return time
    simu = happenTimeSimu(p)
    fig,ax = plt.subplots()
    ax.hist(simu, histtype='step',density=1,bins=range(xlim))
    x = np.array(range(xlim))
    y = p * (np.e ** (-p*x))
    ax.plot(x,y)
    ax.set_xlim([0,xlim])
    plt.show()

happenTimeCompare(0.01,xlim=1000)





kd = 1 #nM
kf = 1e-4 # /nM/s
kr = 1e-4
vol = 100 # ul
NA = 6.02e23 # avagadro

start_conc = np.array([1000,6e23*100e-6*1e-9,0])
start_conc

kd = 1 #nM
kf = 1e-4 # /nM/s
kr = kd*kf
vol = 100 # ul
NA = 6.02e23 # avagadro

start_conc = np.array([100000,6e23*100e-6*1e-9,0])

def stochastic_simu(kd,kf,vol,start_conc,period=30000,plot=0):
    NA = 6.02e23 # avagadro
    kr = kd*kf
    modMatrix = np.array([[-1,-1,1],[1,1,-1]])
    def timestep(start_conc):
        a,t,at = start_conc
        af = kf*a*t/(NA*vol*1e-15)
        ar = kr * at
        suma = af + ar
        r1 = np.random.random()
        dt = 1/(suma) * np.log(1/r1)
        rxn = np.random.choice([0,1],p = [af/suma,ar/suma])
        return start_conc + modMatrix[rxn], dt

    time = [0]
    conc = [start_conc]
    temp = 0
    for i in range(period):
        res,dt = timestep(start_conc)
        start_conc = res
        temp += dt
        if i%1==0:
            conc.append(res)
            time.append(temp)

    fig,ax = plt.subplots()
    ax.plot(np.array(time)/60,[i[plot] for i in conc])
    ax.set_xlabel('Time / minutes')
    ax.set_ylabel('Molecule Number')
    ax.legend(['A','T','AT'])
    ax.grid()
    plt.tight_layout()

import numpy as np
np.random.choice([[1,2],[2,3]],p=[0.1,0.9])
a=np.array([1,2,3])

def test(a):
    a-=1
    print(a)
test(a[-2])
a
THeo = np.random.binomial(1000,0.5)
THeo
A = [i[0] for i in conc]
A[0]
len(A)
fig,ax = plt.subplots()
ax.hist(A, histtype='step',density=1,bins=range(400,600))
ax.hist(THeo,histtype='step',density=1,bins=range(400,600))
plt.show()

vec = np.vectorize(np.random.binomial)
vec([100,100],[0.2,0.5])

a = np.array([1,2,3,4,5.1])
a[a>3] = np.array([1,2])
a
a[a>2]=np.array([0,2,1])
a
a[a<=2] = np.array([44,54])

b = np.zeros_like(a)
b[a>2] = np.array([0,2,1])

b
b[0]=1e-60
b
b=b.astype('float')
b

b =np.array([1,2,3,4,5])
b[b>3] = np.array([1.4,1.43])
b
b
b*=np.array([1.5,1.8,3,4,5])
b

b=b.astype(float)
b[0]=0.000
b = np.array([1,0,0.11,0,13])
b.nonzero()
b
c = np.nonzero(b)

b[b.nonzero()].min()
c
b

b**2
b.sum()

a=1
b=0
if a:
    b=12
elif b:
    print('b')
b
b
a = b[b>100]
a
b
vecbinomial(b,b/10)

vecbinomial = np.vectorize(np.random.binomial,otypes=[float])
# draw by binomial distribution and convert to nM concentration.
scomplex = vecbinomial(a,a)
scomplex
b[b>2]
b[b>2] = np.array([1])
b

b
