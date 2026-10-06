"""Exact MQS coax oracle with a perfectly conducting enclosure return."""
import mpmath as mp
import math
SIGMA=1e6;MU=4e-7*math.pi;A=.005;B=.015;H=.015

def impedance(s,a=A,b=B,h=H,sigma=SIGMA,mu=MU):
    with mp.workdps(max(80,mp.mp.dps)):
        z=mp.mpf(mu)*sigma*s*a*a
        rdc=mp.mpf(h)/(sigma*mp.pi*a*a)
        return rdc*mp.hyper([], [1], z/4)/mp.hyper([], [2], z/4)+s*mu*h*mp.log(b/a)/(2*mp.pi)

def coefficients(s0,count=9,tau=MU*SIGMA*.01**2):
    with mp.workdps(80):return [float(v) for v in mp.taylor(lambda t:impedance(s0+t/tau),mp.mpf(0),count-1)]

def elements(s0,modes=4):
    with mp.workdps(80):
        tau=mp.mpf(MU)*SIGMA*.01**2
        co=mp.taylor(lambda t:impedance(s0+t/tau),mp.mpf(0),2*modes)
        rs=[];ls=[]
        for _ in range(modes):
            rs.append(float(co[0]));co[0]=0
            # Z-R=t L/tau parallel the remaining impedance.
            inverse=[1/co[1]]
            for n in range(1,len(co)-1):inverse.append(-sum(co[j+1]*inverse[n-j] for j in range(1,n+1))/co[1])
            ls.append(float(tau/inverse[0]))
            tail=inverse[1:]
            if len(tail)==0:break
            nextco=[1/tail[0]]
            for n in range(1,len(tail)):nextco.append(-sum(tail[j]*nextco[n-j] for j in range(1,n+1))/tail[0])
            co=nextco
        return rs,ls
