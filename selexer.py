"""
 ███████╗███████╗██╗     ███████╗██╗  ██╗███████╗██████╗
 ██╔════╝██╔════╝██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗
 ███████╗█████╗  ██║     █████╗   ╚███╔╝ █████╗  ██████╔╝
 ╚════██║██╔══╝  ██║     ██╔══╝   ██╔██╗ ██╔══╝  ██╔══██╗
 ███████║███████╗███████╗███████╗██╔╝ ██╗███████╗██║  ██║
 ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
v0.0.1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib import cm
import matplotlib.patches as mpatches

NA = 6.02e23 # avagadro constant

def normalDistribution(avg,std):
    """
    return a normal distribution PDF center around avg, with std
    """
    def cal(x):
        return 1/(std* (2.5066282746310002)) * np.exp(-0.5*((x-avg)/std)**2)
    return cal

def poolBindingTest(pool):
    cc=[]
    per=[]
    kd = pool.meanKd
    for i in np.geomspace(kd/1e3,kd*1e3,20):
        c = i * 1e-9 * 100e-6 * NA
        pool.setupPool(c,rareTreat = 0)
        r = Round(pool,100,kd/1e4)
        _=r.solve_binding_equil(stochasticCutoff = 0)
        cc.append(i)
        per.append(100 * r.targetBindRatio)
    fig,ax = plt.subplots()
    x = np.array(cc) / 1e3
    ax.plot(x,per)
    ax.set_title('Pool binding test with {:.1e}nM Target'.format(kd/1e4))
    ax.set_xscale('log')
    ax.set_xlabel('Pool Conc. / uM')
    ax.set_ylabel('Percent binding %')
    ax.grid()
    plt.tight_layout()
    plt.show()
    return x,per

def plotPoolKdComparison(pools,marker=",",color='tab10'):
    """
    color is one of matplotlib colormap names.
    Colors include:
    """
    cc = """Accent,Blues,BrBG,BuGn,BuPu,CMRmap,Dark2,GnBu,Greens,Greys,OrRd,Oranges
    PRGn,Paired,Pastel1,Pastel2,PiYG,PuBu,PuBuGn,PuOr,PuRd,Purples,RdBu,RdGy
    RdPu,RdYlBu,RdYlGn,Reds,Set1,Set2,Set3,Spectral,Wistia,YlGn,YlGnBu,YlOrBr
    YlOrRd,afmhot,autumn,binary,bone,brg,bwr,cividis,cool,coolwarm,copper,cubehelix
    flag,gist_earth,gist_gray,gist_heat,gist_ncar,gist_rainbow,gist_stern,gist_yarg
    gnuplot,gnuplot2,gray,hot,hsv,inferno,jet,magma,nipy_spectral,ocean,pink
    plasma,prism,rainbow,seismic,spring,summer,tab10,tab20,tab20b,tab20c,terrain
    twilight,twilight_shifted,viridis,winter"""
    cc = [i.strip() for i in cc.split(',')]
    color = cc[color] if isinstance(color,int) else color
    fig,ax = plt.subplots()
    ax.set_title('Pool Kd Histogram')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Kd / nM')
    ax.set_ylabel('Count in Pool')
    leg = []
    for i,p in enumerate(pools):
        alpha = (i+1)/(len(pools)+0.001)
        l = ax.plot(p.kds,p.count/p.count.sum() * 1e13 ,marker,color=cm.get_cmap(color)(alpha),)
        pat = mpatches.Patch(color=l[0].get_color(), label=p.name or f'R{i}')
        leg.append(pat)
    ax.legend(handles=leg)
    plt.tight_layout()
    plt.show()

class Pool:
    """
    represents a pool.
    """
    def __init__(self,kon,koff,frequency,pcrE=2,count=[],name="Lib"):
        """
        kon is an array of pool sequence on rate in nM-1s-1.
        koff is an array of pool sequence off rate in s-1
        kds is an array of pool sequence Kd in nM.
        frequency is the relative frequency of each sequence.
        pcrE is the pcr efficiency of each sequence. between 1 and 2;
        pcrE can also be an array of efficiency for each sequence.
        """
        self.kon = kon
        self.koff = koff
        self.kds = koff/kon
        self.frequency = frequency
        self.pcrE = pcrE
        self.count = count
        self.name = name

    @property
    def meanKd(self):
        """
        this is pool average kd
        """
        return 1/sum(self.frequency/self.kds)

    @property
    def totalcount(self):
        return np.sum(self.count)

    def setupPool(self,totalcount, rareTreat = "chance"):
        """
        give the total copynumber,
        rareTreat: change, or probability or a fraction number as the chance.
        """
        if rareTreat == "chance":
            def func(x):
                return int(x) + np.random.randint(2) * bool(x-int(x))
        elif rareTreat == "probability":
            def func(x):
                return int(x) + (np.random.random() < (x-int(x)))
        else:
            def func(x):
                return int(x) + (np.random.random() < rareTreat) * bool(x-int(x))
        vec = np.vectorize(func)
        count = self.frequency * totalcount
        if count[count.nonzero()].min()<1:
            print('{} Sequence with count < 1 are chosen by {}.'.format(self.name,rareTreat))
        self.count = vec(count).astype(float)
        return self

    def amplify(self,targetcount=None,cycle=None):
        start = self.count.sum()
        if targetcount:
            cycle = 0
            start = self.count.sum()
            while self.count.sum() <targetcount:
                cycle +=1
                self.count = self.pcrE * self.count
        elif cycle:
            self.count = self.count * (self.pcrE ** cycle)
        print('Amplified {} {} cycles, from {:.2e} to {:.2e}'.format(self.name,cycle,start,self.count.sum()))
        self.frequency = self.count / self.count.sum()
        return self


    def plotKdHist(self,marker=',',ax=None):
        if not ax:
            fig,ax = plt.subplots()
        ax.plot(self.kds,self.count,marker)
        ax.set_title(f'Pool {self.name} Kd Histogram')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Kd / nM')
        ax.set_ylabel('Count in Pool')
        plt.tight_layout()
        plt.show()

class NormalDistPool(Pool):
    def __init__(self,bins,kdrange=[1e-3,1e8],koff=1,pcrE=2,kdavg=1e6,kdstd=0.5,name="Lib"):
        """
        Kd in nM,
        normal distribution on Log Concentration scale.
        """
        self.bins = bins
        self.kdrange = kdrange
        self.kdavg = kdavg
        self.kdstd = kdstd
        self.kds = np.geomspace(*kdrange,bins)
        self.kon = koff / self.kds
        self.koff = self.kon * self.kds
        self.pcrE = pcrE
        norm = normalDistribution(np.log10(kdavg),kdstd)
        f = norm(np.log10(self.kds))
        self.frequency = f / f.sum()
        self.name = name

class Round():
    def __init__(self,input,vol,targetConc,incubateTime=120,name=""):
        """
        pool is the initialized selection pool
        vol is volume in uL.
        incubateTime is in minutes
        targetConc is protein concentration in nM.
        """
        self.input = input
        self.inputCount = self.input.count.sum()
        self.vol = vol
        self.incubateTime = incubateTime * 60
        self.targetConc = targetConc
        self.output = None
        self.ns = np.array([])
        self.name = name

    @property
    def outputRatio(self):
        """
        aptamer output / input  ratio.
        """
        return (self.output.sum()+self.ns.sum())/self.inputCount

    @property
    def targetBindRatio(self):
        """
        percentage of target binding to aptamer.
        """
        return self.output.sum() * (1e6 * 1e9 / NA / self.vol) / (self.targetConc)

    def nonspecificBinding(self,nonspecific):
        vec = np.vectorize(np.random.binomial)
        nsbinding = vec(self.input.count,nonspecific)
        self.ns=nsbinding
        return nsbinding

    def _solve_deterministic_ode(self,kon,koff,A0,T0,AT0,EndTime,tol=1e-8):
        """
        solve [A]i + T = [AT]i using ODE.
        kon is the array of [A]i forward rate constant.
        koff is the array of [A]i reverse rate constant.
        A0 is the array of total [A]i concentration.
        T0 is the total T concentration.
        solve by step wise until reach EndTime or reach plateau.
        return timepoints and tracing of AT concentration.
        """
        def binding_ode(y,t,kf,kr,A0,T0):
            "ODE, y is the array of [AT]i conc."
            T = T0 - y.sum()
            return kf*(A0-y)*T - kr*y

        maxKon = kon.max()
        timestep = np.log(2) / (maxKon * T0) / 10 # timestep use 1/10 t(1/2)

        timepoints = np.linspace(0,min(timestep*99,EndTime),100) #min(100000,int(Time/timestep))

        result = odeint(binding_ode,AT0, timepoints, args=(kon,koff,A0, T0))

        while np.absolute((result[-1]-result[-2]) / (timepoints[-1]-timepoints[-2]) ).max() > tol * T0 : # keep integrate
            if timepoints[-1]>= EndTime:
                break
            steepest = np.absolute((result[-1]-result[-2]) / timestep).argmax()
            currentT = T0 - result[-1].sum()
            timestep = np.log(2) / (kon[steepest] * currentT) / 10 # new time step using current T
            newtimepoints = np.linspace( timepoints[-1] , min(timepoints[-1] + timestep*99, EndTime), 100)
            newresult = odeint(binding_ode, result[-1], newtimepoints, args=(kon,koff,A0,T0))
            result = np.append(result, newresult[1:],axis=0)
            timepoints = np.append(timepoints,newtimepoints[1:])

        return timepoints, result



    def solve_binding_ode(self,stochasticCutoff = 1e3):
        pool = self.input
        count = self.input.count
        kds = self.input.kds
        T0 = self.targetConc
        Time = self.incubateTime

        # the aptamer with count > stochasticCutoff will be calculated using derterministic formula.
        dcount = (count > stochasticCutoff)
        dkon = pool.kon[dcount]
        dkoff = pool.koff[dcount]
        dconc = count[dcount]  * (1e6 * 1e9 / NA / self.vol) # conc. in nM

        print('Pool {} binding half life {:.2e} - {:.2e} minutes.'.format(
                    self.name, np.log(2)/60/(dkon.max() * T0), np.log(2) /60/ (dkon.min() * T0) ))

        timepoints, result=self._solve_deterministic_ode(dkon,dkoff,dconc,T0,np.zeros_like(dconc).astype(float),Time)

        self.dTime = timepoints
        self.dATConc = result
        finalTfree = T0 - result[-1].sum()

        # calculate aptame with 0<count < stochasticCutoff using master equation stochastic method.
        scount = np.logical_and(count<=stochasticCutoff,count>0)
        sconc = count[scount]
        skon = pool.kon[scount]
        skoff = pool.koff[scount]
        halflives = np.log(2) / (skon * finalTfree)

        # the one that have halflives < 1/4 of the incubateTime can be considerred reach equilibrium.
        secount = halflives < 0.25 * Time
        sekds = kds[scount][secount]
        sepercentbinding = finalTfree/(finalTfree + sekds)
        vecbinomial = np.vectorize(np.random.binomial,otype=[int])
        secomplex = vecbinomial(sconc[secount],sepercentbinding)

        # the remaining ones use simulation.
        sscount = np.invert(secount)





        # def timestep(start_conc):

        return timepoints, result

    def _solve_stochastic_ode(self,):
        pass 

        # pass
    def _getTfreeAtTime(self,t):
        """
        use ODE solved dTime and dATConc to give Tfree at any given time.
        """
        index = np.searchsorted(self.dTime,t)
        return self.targetConc - self.dATConc[index].sum()




    def solve_binding_equil(self,stochasticCutoff = 1e3):
        count = self.input.count
        kds = self.input.kds
        T0 = self.targetConc

        # the aptamer with count > stochasticCutoff will be calculated using derterministic formula.
        dcount = (count > stochasticCutoff)
        dkds = kds[dcount]
        dconc = count[dcount]  * (1e6 * 1e9 / NA / self.vol) # conc. in nM
        def multiBinding(T,aptConc,aptKd,T0):
            return (T*aptConc/(T+aptKd)).sum() + T - T0
        res = fsolve(multiBinding,T0/2,args=(dconc,dkds,T0))
        Tfree = res[0] # free Target conc. in nM

        daptfree = dkds * dconc / (Tfree + dkds) # calculate free aptamer
        dcomplex = Tfree * daptfree / dkds # calculate aptamer - target complex conc in nM

        # the aptamer with count <= stochasticCutoff will be calculated using binomial distribution
        scount = (count <= stochasticCutoff)
        sconc = count[scount]
        skds = kds[scount]
        percentbinding = Tfree/(skds + Tfree)

        vecbinomial = np.vectorize(np.random.binomial,otypes=[int])
        # draw by binomial distribution and convert to nM concentration.
        scomplex = vecbinomial(sconc,percentbinding)

        resultcount = np.zeros_like(count).astype(float)
        resultcount[dcount] = dcomplex * self.vol * NA / 1e15
        resultcount[scount] = scomplex

        frequency = resultcount / resultcount.sum()
        self.output = resultcount
        return Pool(kon=self.input.kon,koff=self.input.koff,
                frequency = frequency,pcrE=self.input.pcrE,
                count = resultcount)

class Selection():
    def __init__(self,lib,seed=42):
        """
        library is a pool already run setup..
        """
        self.seed = seed
        np.random.seed(seed)
        self.lib = lib
        self.pools = [self.lib]
        self.rounds = []

    def __getitem__(self,i):
        if isinstance(i,int):
            return self.pools[i]
        else:
            return self.pools[int(i[1])]


    def runSelection(self,inputCount,volume,targetConc,
            incubateTime,nonspecific=0,rareTreat='probability',method='equilibrium'):
        """
        selection method can be equilibrium or kinetic
        nonspecific: probability for each sequence to bind nonspecifically.
        """
        pool = self.pools[-1]
        inputCount = eval(inputCount) if isinstance(inputCount,str) else inputCount
        volume = eval(volume) if isinstance(volume,str) else volume
        targetConc = eval(targetConc) if isinstance(targetConc,str) else targetConc
        incubateTime = eval(incubateTime) if isinstance(incubateTime,str) else incubateTime
        nonspecific = eval(nonspecific) if isinstance(nonspecific,str) else nonspecific
        pool.setupPool(inputCount,rareTreat)
        rd = Round(pool,volume,targetConc,incubateTime,name=f'R{1+len(self.rounds)}')
        if method == 'equilibrium':
            nx = rd.solve_binding_equil()
        elif method == 'kinetic':
            nx = rd.solve_binding_ode()

        if nonspecific:
            nsbinding = rd.nonspecificBinding(nonspecific)
            nx.count = nsbinding + nx.count
            nx.frequency = nx.count/nx.count.sum()
        nx.name = f"R{len(self.pools)}"
        self.rounds.append(rd)
        self.pools.append(nx)
        return self

    def amplify(self,targetcount=None,cycle=None):
        _=self.pools[-1].amplify(targetcount=targetcount,cycle=cycle)
        return self

    def setupPool(self,inputCount,rareTreat="chance"):
        self.pools[-1].setupPool(inputCount,rareTreat)
        return self

    def plotPoolsKdHist(self,marker=",",color='Blues',pools=slice(0,None)):
        if isinstance(pools,list):
            toplot = [p for i,p in enumerate(self.pools) if i in pools]
        elif isinstance(pools,int):
            toplot = [self.pools[pools]]
        else:
            toplot = self.pools[pools]
        plotPoolKdComparison(toplot,marker,color)

lib = NormalDistPool(bins=1e4,koff=1,kdrange=[1e-3,1e8],kdavg=1)

lib.setupPool(1e15,rareTreat =1e-2)

rd = Round(lib,100,10,120,name='R1')



t,res = rd.solve_binding_ode()
t[-1]
len(t)
def plot(x):
    fig,ax = plt.subplots()
    ax.plot(t[0:501]/60,[i[x] for i in res[0:501]],)
    ax.set_xlabel('Time / min')
    ax.set_ylabel('Conc. / nM')
    ax.legend(['A','T','AT'])
    ax.grid()
    plt.tight_layout()
    plt.show()
len(t)
len(res[-1])
plot(0)
np.absolute((res[-1]-res[-2])/(t[-1]-t[-2])).max()
len(res[-1])
res[]


fig,ax = plt.subplots()
ax.plot(t[0:501]/60,[10-i.sum() for i in res[0:501]],)
ax.set_xlabel('Time / min')
ax.set_ylabel('Conc. / nM')
ax.legend(['A','T','AT'])
ax.grid()
plt.tight_layout()
plt.show()


a=np.array([[1,2],[3,4]])

np.append(a,np.array([[5,6],[9,8]])[1:],axis=0)

s = Selection(lib,seed=42)

s.runSelection(1e15,100,"pool.meanKd/100000",120,rareTreat=1e-2,nonspecific=1e-5)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100,"pool.meanKd/100000",120,nonspecific=1e-5)\
.amplify(targetcount=1e14)\
.runSelection(1e13,1000,"pool.meanKd/100000",120,nonspecific=1e-5)\
.amplify(targetcount=1e14)\
.runSelection(1e13,10000,"pool.meanKd/100000",120,nonspecific=1e-5)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100000,"pool.meanKd/100000",120,nonspecific=1e-5)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100000,"pool.meanKd/100000",120,nonspecific=1e-5)\
.amplify(targetcount=1e13)\
.setupPool(1e13)


s['R1'].koff

s.plotPoolsKdHist(marker=",",color='tab10',pools=[5,6])

for i in s.pools:
    print('{:.1e}'.format(i.count.sum()))


s.pools[-1].plotKdHist()

i=6
np.logical_and(s[i].count<1000,s[i].count>0).sum()


s.plotPoolsKdHist(marker=".",color='tab10')

for i in s.pools:
    print("{:.2e}".format( i.totalcount))
    print('Pool Kd {:.3g}nM'.format(i.meanKd ))


for i in s.pools:
    print("{:.2e}".format( i.totalcount))
    print('Pool Kd {:.3g}nM'.format(i.meanKd ))

for r in s.rounds:
    print("{:.2e}".format(r.outputRatio))

for r in s.rounds:
    print("{:.2%}".format(r.targetBindRatio))


for r in s.rounds:
    print("{:.2%}".format(r.targetBindRatio))
