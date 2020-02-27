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

def plotPoolKdComparison(pools,marker=",",color='Blues'):
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
        pat = mpatches.Patch(color=l[0].get_color(), label=f'R{i}')
        leg.append(pat)
    ax.legend(handles=leg)
    plt.tight_layout()
    plt.show()

class Pool:
    """
    represents a pool.
    """
    def __init__(self,kon,koff,frequency,pcrE=2,count=[]):
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
            print('Sequence with count < 1 are chosen by {}.'.format(rareTreat))
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
        print('Amplified {} cycles, from {:.2e} to {:.2e}'.format(cycle,start,self.count.sum()))
        self.frequency = self.count / self.count.sum()
        return self


    def plotKdHist(self,marker=',',ax=None):
        if not ax:
            fig,ax = plt.subplots()
        ax.plot(self.kds,self.count,marker)
        ax.set_title('Pool Kd Histogram')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Kd / nM')
        ax.set_ylabel('Count in Pool')
        plt.tight_layout()
        plt.show()



class NormalDistPool(Pool):
    def __init__(self,bins,kdrange=[1e-3,1e8],koff=1,pcrE=2,kdavg=1e6,kdstd=0.5):
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






class Round():
    def __init__(self,input,vol,targetConc,incubateTime=120,):
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

    def solve_binding_ode(self,stochasticCutoff = 1e3):
        pass

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

        # result = np.zeros_like(count).astype(float)
        # result[dcount] = dcomplex
        # result[scount] = scomplex * (1e6 * 1e9 / NA / self.vol)

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
        rd = Round(pool,volume,targetConc,incubateTime)
        if method == 'equilibrium':
            nx = rd.solve_binding_equil()
        elif method == 'kinetic':
            nx = rd.solve_binding_ode()

        if nonspecific:
            nsbinding = rd.nonspecificBinding(nonspecific)
            nx.count = nsbinding + nx.count
            nx.frequency = nx.count/nx.count.sum()

        self.rounds.append(rd)
        self.pools.append(nx)
        return self

    def amplify(self,targetcount=None,cycle=None):
        _=self.pools[-1].amplify(targetcount=targetcount,cycle=cycle)
        return self

    def plotPoolsKdHist(self,marker=",",color='Blues',pools=slice(0,None)):
        plotPoolKdComparison(self.pools[pools],marker,color)

lib = NormalDistPool(bins=1e4,kdrange=[1e-3,1e8],kdavg=1e3)


s = Selection(lib,seed=42)
s.runSelection(1e15,100,"pool.meanKd/100000",120,rareTreat=1e-2)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100,"pool.meanKd/100000",120)\
.amplify(targetcount=1e14)\
.runSelection(1e13,1000,"pool.meanKd/100000",120)\
.amplify(targetcount=1e14)\
.runSelection(1e13,10000,"pool.meanKd/100000",120)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100000,"pool.meanKd/100000",120)\
.amplify(targetcount=1e14)\
.runSelection(1e13,100000,"pool.meanKd/100000",120)\
.amplify(targetcount=1e14)
s.plotPoolsKdHist(marker=".",color='tab10')

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
