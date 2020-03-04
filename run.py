from selexer import Pool, NormalDistPool, Round, Selection,poolBindingTest
import numpy as np



thalf = np.random.random(10000) * (120-1) + 1
lib = NormalDistPool(bins=1e4,kdrange=[1e-3,1e8],koff=np.log(2)/60/thalf,nonspecific=None,
pcrE=2,kdavg=1e5,kdstd=0.7)

lib.setupPool(1e13,rareTreat ='probability')
_=lib.plotPoolHist(xaxis='kds',cumulative=1,xscale='log',yscale='linear',marker="x,",breakpoint=[1])
_.savefig('libcdf.svg')

_=lib.plotPoolHist(xaxis='kon',cumulative=0,xscale='log',yscale='linear')
_.savefig('kon.svg')
_=lib.plotPoolHist(xaxis='koff',cumulative=0,xscale='log',yscale='linear')
_.savefig('koff.svg')

s[1].meanKd
_=poolBindingTest(s[1])
_[0].savefig('R1kd.svg')
lib.meanKd
_[0].savefig('bind.svg')

s = Selection(lib,seed=42)

_1=s['R1'].plotBindingKinetics(index=0,curve='AT',type='s')
_1.savefig('stochasticslow.svg')
_2=s['R1'].plotBindingKinetics(index=-301,curve='AT',type='s')
_2.savefig('stochasticfast.svg')
_.savefig('Tkinetic.svg')

_=s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='kinetic',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)

len(s['R1'].dTime)
len(s['R1'].dATConc)

s['R1']._getTfreeAtTime(12)

_=s.plotPoolsHist(marker="x,",xaxis='kds',cumulative=0,yscale='log',breakpoint=[1])

_.savefig('r1.svg')

a= np.array([[1,2],[2,3],[4,5]])

a.sum(axis=1)
s[2]
s
\


s.Bind(inputCount=1e13,volume=100,targetConc=1,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=0.1,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=0.01,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)
s.Undo(3)
_=s.plotPoolsHist(marker=",",breakpoint=[],xaxis='kds',cumulative=0,yscale='linear')
_.savefig('4round2.png')

_=s.plotPoolsHist(marker=",",xaxis='koff',cumulative=0,yscale='linear',xscale='log')

_=s.plotPoolsHist(marker=",",xaxis='kon',cumulative=0,yscale='linear')


s = Selection(lib,seed=42)

_=s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=60,nonspecific=1e-5,rareTreat=1e-3,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=60,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=60,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=60,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)

_=s.plotPoolsHist(marker="x,",breakpoint=[5],xaxis='kds',cumulative=0,yscale='linear')

_=s.plotPoolsHist(marker=",",xaxis='koff',cumulative=0,yscale='linear',xscale='log')

_=s.plotPoolsHist(marker=",",xaxis='kon',cumulative=0,yscale='linear')




lib = NormalDistPool(bins=1e4,kdrange=[1e-3,1e8],koff=1e-3,nonspecific=None,
pcrE=2,kdavg=1e6,kdstd=0.7)


s = Selection(lib,seed=42)

_=s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-2,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-2,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-2,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-2,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14).plotPoolsHist(marker="x,",breakpoint=[1e3])


s = Selection(lib,seed=42)

s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)




_=s.plotPoolsHist(marker="x,",breakpoint=[1e3])



s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=30,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=30,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=30,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-6,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=30,nsHalflife=10)\
.Amplify(targetcount=1e14)

_=s.plotPoolsHist(marker="x,",breakpoint=[1e3])



s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=5e-3,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14).plotPoolsHist(marker="x,",breakpoint=[1e3])

_=s.plotPoolsHist(marker="x,",breakpoint=[1e3])


_=s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=5e-3,
        method='auto',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)\
.Bind(inputCount=1e13,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='auto')\
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14).plotPoolsHist(marker="x,",breakpoint=[1e3])

_=s.plotPoolsHist(marker="x,",breakpoint=[1e3])





_=s.plotPoolsHist(marker="x,",breakpoint=[1e3],pools=[3])

_=s.plotPoolsHist(marker="x,",breakpoint=[1e3],pools=[4])


lib.setupPool(1e15)

testPool = Pool(kon=np.array([1e-3,1e-3]),koff=np.array([1e-3,1e-2]),frequency=np.array([100,1e13]),name='Test')

testPool.setupPool(1e13)
testPool.count

r = Round(testPool,vol=100,targetConc=10,incubateTime=120)

r.solveBinding(method='kinetic')

_=r.plotBindingKinetics(index=0,curve='AT',type ='s')


r.solveBinding(method='kinetic',stochasticCutoff=1)
_=r.plotBindingKinetics(index=0,curve='AT',type = 'd')
