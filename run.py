from selexer import Pool, NormalDistPool, Round, Selection,poolBindingTest
import numpy as np
from selexer import mprint

mprint.printToScreen = False
#
# testPool = Pool(Thalf=np.array([120,120]),kds=np.array([1,1]),
#                 frequency=np.array([0.5,0.5]),nonspecific=None,pcrE=2,count=np.array([1000,1e9]),name="Test pool")
# testRound = Round(testPool,100,10,incubateTime=120,name="test Round")
#
# for i in range(100):
#
# testRound.solveBindingKinetics()
#
#
#
# for i in range(100):
#     testRound.solveBindingKinetics()
#
#
# for i in range(100):
#     testRound.solveBindingKinetics()
# testRound.plotBindingKinetics(type = 's')


np.random.seed(10)
thalf = np.random.random(10000) * (120-1) + 1
lib = NormalDistPool(bins=1e4,kdrange=[1e-3,1e8],koff=np.log(2)/60/thalf,nonspecific=None,
pcrE=2,kdavg=1e5,kdstd=0.7)
#
s = Selection(lib,seed=42)

_=s.Bind(inputCount=1e15,volume=100,targetConc=10,incubateTime=120,nonspecific=1e-5,rareTreat=1e-3,
        method='kinetic',stochasticCutoff=1e3,seCutoff=1e-1,maxiteration=1e5) \
.Wash(washTime=5,nsHalflife=10)\
.Amplify(targetcount=1e14)


_=s['R1'].plotBindingKinetics(index=325,type='s')
_=s['R1'].plotBindingKinetics(index=325,type='s')


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
