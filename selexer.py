"""
 ███████╗███████╗██╗     ███████╗██╗  ██╗███████╗██████╗
 ██╔════╝██╔════╝██║     ██╔════╝╚██╗██╔╝██╔════╝██╔══██╗
 ███████╗█████╗  ██║     █████╗   ╚███╔╝ █████╗  ██████╔╝
 ╚════██║██╔══╝  ██║     ██╔══╝   ██╔██╗ ██╔══╝  ██╔══██╗
 ███████║███████╗███████╗███████╗██╔╝ ██╗███████╗██║  ██║
 ╚══════╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
v0.0.1
TODO:
1. add different pool distribution.
2. modulize Kd calculaiton
3. enrich score
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from matplotlib import cm
import matplotlib.patches as mpatches
# from mymodule import ft_decorator


NA = 6.02e23 # avagadro constant

def normalDistribution(avg,std):
    """
    return a normal distribution PDF center around avg, with std
    """
    def cal(x):
        return 1/(std* (2.5066282746310002)) * np.exp(-0.5*((x-avg)/std)**2)
    return cal

def poolBindingTest(pool,incubateTime=120,method='equilibrium'):
    """
    mix pool binding test with equilibrium method or kinetics method.
    """
    cc=[]
    per=[]
    kd = pool.meanKd
    for k,i in enumerate(np.geomspace(kd/1e3,kd*1e3,21)):
        c = i * 1e-9 * 100e-6 * NA
        pool.setupPool(c,rareTreat = 0)
        r = Round(pool,100,kd/1e4,incubateTime,f'Test {k+1}')
        _=r.solveBinding(method,stochasticCutoff = 0)
        cc.append(i)
        per.append(100 * sum(r.targetBindRatio.values()))
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
    return fig,x,per

def plotPoolHist(pools,xaxis='kds',marker=",",color='tab10',breakpoint=[],cumulative=False,**kwargs):
    """
    xaxis can be any property of the pool. [kds,kon,koff,pcrE,nonspecific]
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
    curve = xaxis.capitalize()
    cc = [i.strip() for i in cc.split(',')]
    color = cc[color] if isinstance(color,int) else color
    fig,ax = plt.subplots()
    ax.set_title(f'Pool {curve} Histogram {"CDF" * cumulative}')
    ax.set_xscale(kwargs.get('xscale','log'))
    ax.set_yscale(kwargs.get('yscale','log'))
    unit = "nM" if curve=="Kds" else ("s-1" if curve=='Koff' else "nM-1s-1")
    ax.set_xlabel(f'{curve} / {unit}')
    ax.set_ylabel('Percentage in Pool %')
    leg = []
    for i,p in enumerate(pools):
        alpha = (i+1)/(len(pools)+0.001)
        ccolor=cm.get_cmap(color)(alpha)
        toplotX,toplotY = getattr(p,xaxis),p.count/p.count.sum() * 100

        for k,c,m in segmentArray(toplotX,toplotY,marker,breakpoint,cumulative):
            l = ax.plot(k,c ,m,color=ccolor,)
        pat = mpatches.Patch(color=l[0].get_color(), label=p.name or f'R{i}')
        leg.append(pat)
    ax.legend(handles=leg,bbox_to_anchor=(1, 0.8))
    plt.tight_layout()
    plt.show()
    return fig

def segmentArray(kds,counts,marker,breakpoint,cumulative):
    """
    sort the array first,
    then segment it by breakpoint. return array without zero.
    """
    index = 0
    bindex = 0
    order = kds.argsort()
    kds = kds[order]
    counts = counts[order]
    if cumulative:
        counts = np.cumsum(counts)
    for i in breakpoint:
        new = np.absolute(kds[bindex:] -i ).argmin() + bindex
        k,c = removeZero( kds[bindex:new],counts[bindex:new])
        yield k,c,marker[index]
        bindex = new
        index +=1
    k,c = removeZero( kds[bindex:],counts[bindex:])

    yield k,c,marker[index]

def removeZero(kd,count):
    return kd[count.nonzero()],count[count.nonzero()]

def calculateLoseChance(scount,skds,Tfree,meanKd):
    """
    Given a count , kds, array, and Tfree, calculate chance to loose
    in a binomial distribution.
    """

    if len(skds) == 0: return 0
    bestkds = np.where(scount>0,skds,skds.max())
    idx = bestkds.argmin()
    c = scount[idx]
    k = bestkds[idx]
    if k>=meanKd: return 0 # if rare sequence have bad kd, no chance to lose good ones
    p = Tfree / (Tfree + k)
    return (1-p)**(int(c))

class Pool:
    """
    represents a pool.
    newly generated property:
    self.amplifycycle: cycles amplified.
    self.nscount: count that is from nonspecific binding, after wash, the leftover of nsbinding.
    self.sbcount: after wash the leftover count that is from specific binding.
    """
    def __init__(self,kon=None,koff=None,Thalf=None,kds=None,
                    frequency=None,nonspecific=None,pcrE=2,count=[],name="Lib"):
        """
        kon is an array of pool sequence on rate in nM-1s-1.
        koff is an array of pool sequence off rate in s-1
        kds is an array of pool sequence Kd in nM.
        Thalf is pool off half life in minutes.
        frequency is the relative frequency of each sequence.
        nonspecific is the relative nonspecific binding ability. default is 1.
        2 is 2 fold more likely to bind; 0.5 is half.
        pcrE is the pcr efficiency of each sequence. between 1 and 2;
        pcrE can also be an array of efficiency for each sequence.
        """
        self.kon = kon
        self.koff = koff
        self.kds = kds
        if Thalf:
            self.koff = np.log(2) / Thalf / 60
        if self.kon is None:
            self.kon = self.koff / self.kds
        elif self.koff is None:
            self.koff = self.kon * self.kds
        elif self.kds is None:
            self.kds = self.koff/self.kon
        else:
            raise ValueError ('Need input to determine kinetic rate.')

        self.frequency = frequency
        if self.frequency is None:
            self.frequency = np.full(len(self.kds),1/len(self.kds))
        if self.frequency.sum() != 1.0:
            self.frequency = self.frequency / self.frequency.sum()
        self.nonspecific = nonspecific
        if nonspecific is None:
            self.nonspecific = np.full(len(self.kds),1)

        self.pcrE = pcrE
        self.count = count
        self.name = name

    def __repr__(self):
        return f"Pool {self.name}"

    @property
    def meanKd(self):
        """
        this is pool average kd
        """
        return 1/(self.frequency/self.kds).sum()

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
            print('{} Sequence with count < 1 are chosen by {}.'.format(self,rareTreat))
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
        print('{}  amplified {} cycles, from {:.2e} to {:.2e}'.format(self,cycle,start,self.count.sum()))
        self.amplifycycle = cycle
        self.frequency = self.count / self.count.sum()
        return self

    def wash(self,washTime,nsOffrate=None,nsHalflife=None):
        """
        washTime: in minutes
        nsHalflife : in minutes
        WIll update self.nscount and create self.sbcount.
        """
        washTime = washTime * 60
        vec = np.vectorize(np.random.binomial)
        # calculate specific binding left over.
        sb = self.count - self.nscount
        sbleftper = np.exp(- self.koff * washTime)
        self.sbcount = vec(sb,sbleftper) # the left amout follow binomial distribution.

        # calculate ns binding left over.
        if nsHalflife is not None:
            nsleftper = (0.5)**( washTime / 60.0 / nsHalflife)
        elif nsOffrate is not None:
            nsleftper = np.exp(-nsOffrate * washTime)
        else:
            nsleftper = 1
        self.nscount = vec(self.nscount, nsleftper)
        self.count = self.sbcount + self.nscount
        if not self.count.sum():
            raise ValueError ('Wash too harsh. Nonthing left.')
        self.frequency = self.count/self.count.sum()
        return self

    def plotPoolHist(self,xaxis='kds',marker=',',color='tab10',cumulative=False,**kwargs):
        return plotPoolHist([self],xaxis=xaxis,cumulative=cumulative,marker=marker,**kwargs)

class NormalDistPool(Pool):
    def __init__(self,bins,kdrange=[1e-3,1e8],koff=1,nonspecific=None,pcrE=2,kdavg=1e6,kdstd=0.5,name="Lib",):
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
        self.nonspecific = nonspecific
        if nonspecific is None:
            self.nonspecific = np.full(len(self.kds),1)

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
        self.equilibriumOutput = np.array([])
        self.kineticsOutput = np.array([])
        self.ns = np.array([]) # nonspecific binding of this round.
        self.name = name

    def __repr__(self):
        return f"Round {self.name}"

    @property
    def aptamerConc(self):
        """
        return round aptamer concentration in nM.
        """
        return self.inputCount/self.vol/NA*1e9*1e6

    @property
    def outputRatio(self,):
        """
        aptamer output / input  ratio.
        """
        e = bool(self.equilibriumOutput.sum()) and (self.equilibriumOutput.sum()+self.ns.sum())/self.inputCount
        k = bool(self.kineticsOutput.sum()) and (self.kineticsOutput.sum()+self.ns.sum())/self.inputCount
        return {'equilibrium':e,'kinetics':k}

    @property
    def targetBindRatio(self):
        """
        percentage of target binding to aptamer.
        """
        e= self.equilibriumOutput.sum() * (1e6 * 1e9 / NA / self.vol) / (self.targetConc) or False
        k = self.kineticsOutput.sum() * (1e6 * 1e9 / NA / self.vol) / (self.targetConc) or False
        return  {'equilibrium':e,'kinetics':k}

    def nonspecificBinding(self,nonspecific):
        """
        determine the nonspecific binding of this round.
        nonspecific is an array or float, represents fraction of
        an aptamer nonspecifically binds.
        """
        vec = np.vectorize(np.random.binomial)
        nsbinding = vec(self.input.count,nonspecific*self.input.nonspecific)
        self.ns=nsbinding
        return nsbinding

    def plotBindingKinetics(self,index=0,curve='AT', type = 'd'):
        """
        type is [d]eterministic or [s]tochastic
        """
        p = self.input
        def plot(x,y,title,ylabel='Conc. / nM'):
            fig,ax = plt.subplots()
            ax.plot(x,y)
            ax.set_xlabel('Time / min')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.title.set_position([0.5, 1.05])
            ax.grid()
            plt.tight_layout()
            plt.show()
            return fig
        if type in ['deterministic','d']:
            assert index < len(self.dIndex), (f"Max deterministic index {len(self.dIndex)-1}")
            a = self.dIndex[index]
            title = "Kd {:.1e}nM; Kon {:.1e}nMs-1; Koff {:.1e}s-1".format(
                p.kds[a],p.kon[a],p.koff[a] )
            if curve == 'AT':
                y = self.dATConc[:,index]
            elif curve == 'A':
                aconc = self.input.count[a]  * (1e6 * 1e9 / NA / self.vol)
                y = aconc - self.dATConc[:,index]
            elif curve == 'T':
                y = self.targetConc - self.dATConc.sum(axis=1)
                title = f"Target Starting Conc. {self.targetConc}nM"
            ylabel = f"{curve} Conc. /nM"
            return plot( self.dTime /60 , y ,title, ylabel)
        elif type in ['stochastic','s']:
            assert index < len(self.ssIndex), (f"Max stochastic index {len(self.ssIndex)-1}")
            a = self.ssIndex[index]
            title = "Kd {:.1e}nM; Kon {:.1e}nM-1s-1; Koff {:.1e}s-1".format(
                p.kds[a],p.kon[a],p.koff[a] )
            t = self.ssTrace[index][0]
            if curve == 'AT':
                y = self.ssTrace[index][1]
            elif curve == 'A':
                aconc = self.input.count[a]
                y = aconc - self.ssTrace[index][1]
            elif curve == 'T':
                t = self.dTime
                y = self.targetConc - self.dATconc.sum(axis=1)
                title = f"Target Starting Conc. {self.targetConc}nM"
            return plot( t/60 , y ,title,ylabel=f"[{curve}] Count")

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

        # keep integrate while the slope of curve is over tolerance * T0 or time hits EndTime
        while np.absolute((result[-1]-result[-2]) / (timepoints[-1]-timepoints[-2]) ).max() > tol * T0 :
            if timepoints[-1]>= EndTime:
                break
            # calculate following timestep by kobs of the steepest curve.
            steepest = np.absolute((result[-1]-result[-2]) / timestep).argmax()
            currentT = T0 - result[-1].sum()
            timestep = np.log(2) / (kon[steepest] * currentT) / 10 # new time step using current T

            # calculate following time step by
            # steepest = np.absolute((result[-1]-result[-2]) / timestep).max()
            # timestep = min(1, 1 * T0 / steepest )

            newtimepoints = np.linspace( timepoints[-1] , min(timepoints[-1] + timestep*99, EndTime), 100)
            newresult = odeint(binding_ode, result[-1], newtimepoints, args=(kon,koff,A0,T0))
            result = np.append(result, newresult[1:],axis=0)
            timepoints = np.append(timepoints,newtimepoints[1:])

        return timepoints, result

    def _solve_stochastic_ode(self,kon,koff,conc,maxiteration=1e4):
        """
        solve time course of stochastic binding of aptamer in this round context.
        """
        EndTime = self.incubateTime
        time = [0]
        ATconc = [0] # record of AT complex count
        temp = 0 # keep track of time
        cycles = 0
        index = 0
        dTime = self.dTime
        dTimelength = len(dTime)-1
        freeT = self.targetConc - self.dATConc.sum(axis=1)

        while cycles < maxiteration:
            cycles +=1
            a = conc - ATconc[-1] # input is count.

            # t = self._getTfreeAtTime(temp)  # get free Target conc. in nM

            while dTime[index] < temp:
                if index == dTimelength:
                    break
                else:
                    index +=1
            t = freeT[index]
            # if dTime[index]>=temp:
            #     t = freeT[index]
            # else:
            #     if index == dTimelength:
            #         t=freeT[-1]
            #     else:
            #         index +=1


            ar = koff * ATconc[-1]
            af = kon * a * t
            suma = af + ar
            r1 = np.random.random()
            dt = 1/suma * np.log(1/r1)
            rxn = np.random.choice([1,-1],p=[af/suma,ar/suma]) # choose reaction forward or reverse.
            temp += dt
            if temp > EndTime:
                break
            ATconc.append(ATconc[-1]+rxn) # update AT count record.
            time.append(temp)
        return np.array(time),np.array(ATconc)

    def _getTfreeAtTime(self,t):
        """
        use ODE solved dTime and dATConc to give Tfree at any given time.
        """
        index = np.absolute(self.dTime - t).argmin()
        return self.targetConc - self.dATConc[index].sum()

    def solveBindingKinetics(self,stochasticCutoff = 1e3,seCutoff = 1e-2,maxiteration=1e4,**kwargs):
        """
        stochasticCutoff: how many counts down to use stochastic method.
        seCutoff: halflives cut off for a sequence to be considerred by stochastic method.
        """
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

        if len(dkon)>0:
            print('{} binding half life {:.2e} ~ {:.2e} minutes.'.format(
                        self.input, np.log(2)/60/(dkon.max() * T0), np.log(2) /60/ (dkon.min() * T0) ))

        timepoints, result=self._solve_deterministic_ode(dkon,dkoff,dconc,T0,np.zeros_like(dconc).astype(float),Time)

        self.dIndex = dcount.nonzero()[0]
        self.dTime = timepoints
        self.dATConc = result

        Tfree = T0 - result[-1].sum()

        # calculate aptame with 0<count < stochasticCutoff using master equation stochastic method.
        scount = np.logical_and(count<=stochasticCutoff,count>0)
        sconc = count[scount]
        skon = pool.kon[scount]
        skoff = pool.koff[scount]
        halflives = np.log(2) / (skon * Tfree)
        skds = kds[scount]


        print('{} lose best binder in pool chance: {:.2%}'.format(self, calculateLoseChance(sconc,skds,Tfree,self.input.meanKd)))
        # the one that have halflives < 1/10 of the incubateTime can be considerred reach equilibrium.
        secount = halflives < seCutoff * Time
        print('{} fast stochastic count {}'.format(self.input, secount.sum()))
        sekds = skds[secount]
        sepercentbinding = Tfree/(Tfree + sekds)
        vecbinomial = np.vectorize(np.random.binomial,otypes=[int])
        secomplex = vecbinomial(sconc[secount],sepercentbinding)

        # the remaining ones use simulation.
        sscount = np.invert(secount)
        print('{} slow stochastic count {}'.format(self.input, sscount.sum()))
        ssconc = sconc[sscount]
        sskon = skon[sscount]
        sskoff = skoff[sscount]
        self.ssTrace = []
        self.ssIndex = scount.nonzero()[0][sscount]
        sscomplex = []
        for on,off,c in zip(sskon,sskoff,ssconc):
            t,atconc = self._solve_stochastic_ode(on,off,c,maxiteration=maxiteration)
            self.ssTrace.append((t,atconc))
            sscomplex.append(atconc[-1])

        # combine results together
        resultcount = np.zeros_like(count).astype(float)
        resultcount[dcount] = self.dATConc[-1] * self.vol * NA / 1e15
        temps = resultcount[scount]
        temps[secount] = secomplex
        temps[sscount] = np.array(sscomplex)
        resultcount[scount] = temps

        frequency = resultcount / resultcount.sum()
        self.kineticsOutput = resultcount
        return Pool(kon=self.input.kon,koff=self.input.koff,
                frequency = frequency,pcrE=self.input.pcrE,
                count = resultcount,name=self.name)

    def solveBindingEquilibrium(self,stochasticCutoff = 1e3,**kwargs):
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

        # calculate the chance to loose best binder in pool.

        print('{} lose best binder in pool chance: {:.2%}'.format(self, calculateLoseChance(sconc,skds,Tfree,self.input.meanKd)))




        resultcount = np.zeros_like(count).astype(float)
        resultcount[dcount] = dcomplex * self.vol * NA / 1e15
        resultcount[scount] = scomplex

        frequency = resultcount / resultcount.sum()
        self.equilibriumOutput = resultcount
        return Pool(kon=self.input.kon,koff=self.input.koff,
                frequency = frequency,pcrE=self.input.pcrE,
                count = resultcount,name=self.name)

    def solveBinding(self,method='auto',**kwargs):
        """
        method can be equilibrium or kinetic or auto
        kwargs are additional arguments
        for equilibrium: stochasticCutoff=1e3
        for kinetic: stochasticCutoff = 1e3, seCutoff = 1e-2, maxiteration=1e4,
        for auto, determine to use which one based on kobs estimate.
        """

        if method == 'equilibrium':
            return self.solveBindingEquilibrium(**kwargs)
        elif method == 'kinetic':
            return self.solveBindingKinetics(**kwargs)
        elif method == 'auto':
            kon = self.input.kon[self.input.count>0]
            freq = self.input.frequency[self.input.count>0]
            # thalf = np.log(2)/(kon * self.targetConc).min() / 60
            thalf = np.log(2)/((1/(freq/kon).sum()) * max( self.targetConc, self.aptamerConc)) / 60

            if self.incubateTime > thalf * 5:
                print(f"{self} solve binding using Equilibrium method")
                return self.solveBindingEquilibrium(**kwargs)
            else:
                print(f"{self} solve binding using Kinetics method")
                return self.solveBindingKinetics(**kwargs)
        else:
            print('method not exist.')
            return 0


class Selection():
    def __init__(self,lib,seed=42):
        """
        library is a pool already run setup..
        """
        self.seed = [seed]
        # np.random.seed(seed)
        self.lib = lib
        self.pools = [self.lib]
        self.rounds = []

    def __repr__(self):
        return f"Selection {len(self.rounds)} rounds"

    def __getitem__(self,i):
        if isinstance(i,int):
            return self.pools[i]
        else:
            if i.startswith('P'):
                return self.pools[int(i[1:])]
            elif i.startswith('R'):
                return self.rounds[int(i[1:])-1]

    def Undo(self,n=1):
        """
        remove last round and pool in the selection.
        """
        for i in range(n):
            _=self.pools.pop()
            _=self.rounds.pop()
            _=self.seed.pop()
        return self

    def Bind(self,inputCount,volume,targetConc,
            incubateTime,nonspecific=0,rareTreat='probability',
            **kwargs):
        """
        incubateTime : in minutes
        nonspecific: probability for each sequence to bind nonspecifically.
        kwargs: additional arguments to solveBinding.
        Including:
        method can be equilibrium or kinetic or 'auto'; default is auto.
        for equilibrium: stochasticCutoff=1e3
        for kinetic: stochasticCutoff = 1e3, seCutoff = 1e-2, maxiteration=1e4,
        """
        np.random.seed(self.seed[-1])
        self.seed.append(np.random.randint(1,1000))
        pool = self.pools[-1]
        inputCount = eval(inputCount) if isinstance(inputCount,str) else inputCount
        volume = eval(volume) if isinstance(volume,str) else volume
        targetConc = eval(targetConc) if isinstance(targetConc,str) else targetConc
        incubateTime = eval(incubateTime) if isinstance(incubateTime,str) else incubateTime
        nonspecific = eval(nonspecific) if isinstance(nonspecific,str) else nonspecific
        pool.setupPool(inputCount,rareTreat)
        rd = Round(pool,volume,targetConc,incubateTime,name=f'R{1+len(self.rounds)}')

        nx = rd.solveBinding(**kwargs) # generate a new pool from Round.solveBinding

        if nonspecific:
            nsbinding = rd.nonspecificBinding(nonspecific)
            nx.count = nsbinding + nx.count
            nx.nscount = nsbinding
            nx.frequency = nx.count/nx.count.sum()
        self.rounds.append(rd)
        self.pools.append(nx)
        return self

    def Wash(self,washTime,nsOffrate=None,nsHalflife=None):
        """
        wash current round.
        nsOffrate is the off rate for ns binding be removed.
        """
        np.random.seed(self.seed[-1])
        _=self.pools[-1].wash(washTime,nsOffrate,nsHalflife)
        return self

    def Amplify(self,targetcount=None,cycle=None):
        np.random.seed(self.seed[-1])
        _=self.pools[-1].amplify(targetcount=targetcount,cycle=cycle)
        return self

    def setupPool(self,inputCount,rareTreat="chance"):
        self.pools[-1].setupPool(inputCount,rareTreat)
        return self

    def plotPoolsHist(self,xaxis='kds',marker=",",color='tab10',breakpoint=[],
                    pools=slice(0,None),cumulative=False,**kwargs):
        if isinstance(pools,list):
            toplot = [p for i,p in enumerate(self.pools) if i in pools]
        elif isinstance(pools,int):
            toplot = [self.pools[pools]]
        else:
            toplot = self.pools[pools]
        return plotPoolHist(toplot,xaxis,marker,color,breakpoint,cumulative=cumulative,**kwargs)
