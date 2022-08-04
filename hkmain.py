# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:48:19 2022

@author: Hans

visualizing R1r work for an upcoming paper

"""
import RDmath2 as RDmath
import numpy as np
from matplotlib import pyplot as plt

""" Palmer group - color-blindness-friendly color, and 
BW print-friendly linestyle definitions"""

reddishpurple = [204/255.0,121/255.0,167/255.0]
green = [0,158/255.0,115/255.0]
orange = [230/255.0,159/255.0,0/255.0]
blue = [0/255.0,114/255.0,178/255.0]

linestyles_dict = {
     'loosely dotted' :        (0, (1, 10)),
     'dotted' :                (0, (1, 1)),
     'densely dotted' :        (0, (1, 1)),
     'loosely dashed' :        (0, (5, 10)),
     'dashed' :                (0, (5, 5)),
     'densely dashed' :        (0, (5, 1)),
     'loosely dashdotted' :    (0, (3, 10, 1, 10)),
     'dashdotted' :            (0, (3, 5, 1, 5)),
     'densely dashdotted' :    (0, (3, 1, 1, 1)),
     'dashdotdotted' :         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted' : (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted' : (0, (3, 1, 1, 1, 1, 1))
}


"""2-site example for R1rho with variable w1 field which is stepped in sync with the rf position"""

omA = 0
omB = 2*np.pi*200
rf0 = np.pi*200
dw = np.abs(omB-omA)

R1A = 1
R2A= 20
R1B = 1
R2B = 20
pA = 0.90
pB = 1-pA
kex = 0.1*dw
k21 = kex*pA
k12 = kex*pB

numbv=200
w10 = 0.5*dw

rfa = np.linspace(1,2*dw,numbv)

ombar0 = pA*omA + pB*omB
tanthet0 = w10/(ombar0-rf0)
ombar = pA*(omA-ombar0-rfa) + pB*(omB-ombar0-rfa)
w1a = np.abs(tanthet0*(ombar))
we = np.sqrt(ombar**2 + w1a**2)
sinthet = w1a/we
costhet = ombar/we


plt.figure(9)
plt.clf()

R1rInt0 = (RDmath.Laguerre(omA-rfa-ombar0,R1A,R2A,omB-rfa-ombar0,R1B,R2B,w1a,k12*2.5,k21*2.5))
res=[RDmath.rexeq(i+ombar0,dw,pB,k12*2.5+k21*2.5,2,w1a[j],0,0,0,0,0,0,0,0,0,0,0) for j,i in enumerate(rfa)]    ######

plt.plot((w1a)/(2*np.pi),R1rInt0,c='red')
plt.plot((w1a)/(2*np.pi),RDmath.R1r_n(res,sinthet,costhet,[pA,pB],[R1A,R1B],[R2A,R2B]),c='cyan')

plt.ylim([0,80])

#%%

def getmidexp(omA,omB,omC,k12ex,k13ex,k23ex,pB,pC,w10,rf0,dw,dwc,R1,R2):
    """N-site middle experiment work-in-progrerss"""
    ombar0 = pA*omA + pB*omB+ pC*omC
    tanthet0 = w10/(ombar0-rf0)
    rfa = np.linspace(60,2*np.max([omB,omC]),200)
    ombar = pA*(omA-rfa-ombar0) + pB*(omB-rfa-ombar0)+ pC*(omC-rfa-ombar0)
    w1a = np.abs(tanthet0*(ombar))
    we = np.sqrt(ombar**2 + w1a**2)
    sinthet = w1a/we
    costhet = ombar/we
    rexaltc=np.array([RDmath.rexeq(i+ombar0,dw+i+ombar0,pB,k12ex,2,w1a[j],0,0,dwc+i+ombar0,pC,k13ex,k23ex,0,0,0,0,0) for j,i in enumerate(rfa)])
    x=(w1a)/(2*np.pi)
    y=RDmath.R1r_n(rexaltc,sinthet,costhet,[pA,pB,pC],R1,R2)
    return x,y



#%%
#own example from Structure paper


shc=[[124.51,122.5,122.3],[117.99,118.2,118.5],[115.4,115.3,115.1],[126.3,126.1,126.1],[115.03,117.2,117.3],[0,5.53,1.38],[0,3.52,1.91],[0,0.32,-2.91]]
shn=[30,32,45,52,78,38,77,73]
R1 = [1,1,1]
R2= [20,20,20]
R1B = 1
R2B = 20
pB = 0.0156
pC = 0.0156
pA = 1-pB-pC
k12ex = 3150
k13ex = 695
omA = 0
plt.figure(10)
plt.clf()
sel=0
omB = np.abs(shc[sel][0]-shc[sel][2])*np.pi*2*50
omC = np.abs(shc[sel][0]-shc[sel][1])*np.pi*2*50
print(shn[sel],omB,omC,np.abs(shc[sel][0]-shc[sel][1]), np.abs(shc[sel][0]-shc[sel][2]))
dw = np.abs(omB-omA)
dwc = np.abs(omC-omA)
if sel == 7:
    dwbc = np.abs(omC+omB)
    w10 = 0.5*(omC+omB)
else:
    dwbc = np.abs(omC-omB)
    w10 = 0.5*dwbc#np.average([dw])
#w10=dw*0.5
#print w10
rf0=np.min([shc[sel][2],shc[sel][1]])*np.pi*2*0+w10
#k23ex = 
#print(omA,omB,omC,k12ex,k13ex,1,pB,pC,w10,rf0)

ver=1
maxc=[];maxax=[];minax=[]

if ver == 1:
    x,y=getmidexp(omA,omB,omC,k12ex,k13ex,1,pB,pC,w10,rf0,dw,dwc,R1,R2)
    plt.plot(x,y,c=blue,linestyle='dashed',label='B0=500 MHz, no minor site exchange')
else:
    x,y=getmidexp(omA*80/50,omB*80/50,omC*80/50,k12ex,k13ex,1,pB,pC,w10*80/50,rf0*80/50,dw*80/50,dwc*80/50,R1,R2)
    plt.plot(x,y,c=blue,linestyle='solid',label='B0=500 MHz, no minor site exch')
maxc.append(x[int(np.argmax(y))])
maxax.append(np.max(y))
minax.append(np.min(y))
#print omA,omB,omC,k12ex,k13ex,1,pB,pC,w10,rf0
if ver == 1:
    x,y=getmidexp(omA,omB,omC,3500,315,860,pB,pC,w10,rf0,dw,dwc,R1,R2)
    plt.plot(x,y,c=orange,linestyle='dashed',label='B0=500 MHz, triangular as in Fig. S13A')
else:
    x,y=getmidexp(omA*80/50,omB*80/50,omC*80/50,k12ex,k13ex,100,pB,pC,w10*80/50,rf0*80/50,dw*80/50,dwc*80/50,R1,R2)
    plt.plot(x,y,c=blue,linestyle='dashed',label='B0=500 MHz, 100 s-1 minor site')
maxc.append(x[int(np.argmax(y))])#print(omA*120/50,omB*120/50,omC*120/50,k12ex,k13ex,1,pB,pC,w10*120/50,rf0*120/50)
maxax.append(np.max(y))
minax.append(np.min(y))
#print(x,y)
if ver == 1:
    x,y=getmidexp(omA*120/50,omB*120/50,omC*120/50,k12ex,k13ex,1,pB,pC,w10*120/50,rf0*120/50,dw*120/50,dwc*120/50,R1,R2)
    plt.plot(x,y,c=blue,label='B0=1200 MHz, no minor site exchange')
else:
    x,y=getmidexp(omA*80/50,omB*80/50,omC*80/50,k12ex,k13ex,1000,pB,pC,w10*80/50,rf0*80/50,dw*80/50,dwc*80/50,R1,R2)
    plt.plot(x,y,c=blue,linestyle='dashdot',label='B0=500 MHz, 1000 s-1 minor site')
maxc.append(x[int(np.argmax(y))])
maxax.append(np.max(y))
minax.append(np.min(y))
#print omA,omB,omC,k12ex,k13ex,1,pB,pC,w10,rf0
#x,y=getmidexp(omA*120/50,omB*120/50,omC*120/50,k12ex,k13ex,860,pB,pC,w10*120/50,rf0*120/50,dw*120/50,dwc*120/50)
if ver == 1:
    x,y=getmidexp(omA*120/50,omB*120/50,omC*120/50,3500,315,860,pB,pC,w10*120/50,rf0*120/50,dw*120/50,dwc*120/50,R1,R2)
    plt.plot(x,y,c=orange,label='B0=1200 MHz, triangular as in Fig. S13A')
else:
    x,y=getmidexp(omA*80/50,omB*80/50,omC*80/50,k12ex,k13ex,10000,pB,pC,w10*80/50,rf0*80/50,dw*80/50,dwc*80/50,R1,R2)
    plt.plot(x,y,c=blue,linestyle='dotted',label='B0=500 MHz, 10000 s-1 minor site')
maxc.append(x[int(np.argmax(y))])
maxax.append(np.max(y))
minax.append(np.min(y))
#plt.plot((w1a)/(2*np.pi),R1r(rexaltc2,sinthet0,costhet0,pA,pB,R2A,R2B,R1A,R1B),c='cyan')
#argpos=int(np.argmax(R1r(rexaltc,sinthet0,costhet0,pA,pB,R2A,R2B,R1A,R1B)))
for i in maxc:
    plt.plot([i,i],[np.min(minax),np.max(maxax)],linestyle='dotted',c='black')

print(np.min(minax),((np.max(maxax)-np.min(minax))*0.05))
plt.ylim([np.min(minax)-((np.max(maxax)-np.min(minax))*0.05),np.max(maxax)+(np.max(maxax)-np.min(minax))*0.05])
plt.title('residue '+str(shn[sel]))
plt.legend()
plt.show()
plt.xlabel(r'$\omega_1 / 2\pi$')
plt.ylabel(r'$R_{1\rho}$ (1/s)',fontsize=10)
#plt.set_yticklabels([])
#ax11.tick_params(axis='both', which='major', labelsize=8)
#ax11.set_xlabel(r'$\Omega/\omega_1$')
#ax11.text(0.03,0.05,"(k)",transform=ax11.transAxes,fontsize=8)

#%%
[[  -31.41592654 -2513.27412287     0.           282.74333882
      0.             0.        ]
 [ 2513.27412287   -31.41592654  3141.59265359     0.
    282.74333882     0.        ]
 [    0.         -3141.59265359   -31.41592654     0.
      0.           282.74333882]
 [   31.41592654     0.             0.          -282.74333882
  -1256.63706144     0.        ]
 [    0.            31.41592654     0.          1256.63706144
   -282.74333882  3141.59265359]
 [    0.             0.            31.41592654     0.
  -3141.59265359  -282.74333882]] LKM

#%%
.0755270527520413 x
0.07473124175743817 x
0.07394761102510498 x
0.07317592028644058 x
0.07241593498123092 x
0.07166742610166514 x
0.07093017004107563 x
0.07020394844724967 x
0.0694885480801639 x
0.06878376067399918 x
0.06808938280329768 x
0.06740521575312818 x
0.06673106539313073 x
0.06606674205531571 x
0.06541206041549627 x
0.06476683937823832 x
#%%

indx=int(np.argmax(R1rInt0))

r1r=R1rInt0
rexx=Rex(r1r,sinthet,costhet,pA,pB,R2A,R2B,R1A,R1B)
plt.plot((w1a)/(2*np.pi),r1r,c='red',ls='dashed')
plt.show()




#rexxc=Rex(r1ralt3,sinthet,costhet,pA,pB,R2A,R2B,R1A,R1B)
#print(rexxa)
#print(rexxb)
#print(rexxb)
#print(rexxc)
#%%
print(rexx)
plt.plot((rfa)/(2*np.pi),rexx,c=blue,ls='dashed')
plt.show()



#%%

"""plot result"""
plt.figure(14)
plt.clf()
r1r=res
rexx=Rex(r1r,sinthet,costhet,pA,pB,R2A,R2B,R1A,R1B)
r1rr1r=R1r(r1r,sinthet,costhet,pA,pB,R2A,R2B,R1A,R1B)

#cestx=cest(r1r,0.5,sinthet,costhet,pA,pB,R2A,R2B,R1A,R1B)
plt.plot((rfa)/(2*np.pi),r1r,c=blue,ls='solid')
#plt.plot((rfa)/(2*np.pi),costhet**2,c=blue,ls='dashed')

#plt.plot((rfb)/(2*np.pi),sinthet**2,c=blue,ls='dashed')
plt.show()

