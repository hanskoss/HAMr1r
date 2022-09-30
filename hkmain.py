# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 11:48:19 2022

@author: Hans

visualizing R1r work for an upcoming paper

"""
import RDmath2 as RDmath
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eig
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
     'dashdotted2' :            (0, (5, 3, 1, 3)),
     'densely dashdotted' :    (0, (3, 1, 1, 1)),
     'dashdotdotted' :         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted' : (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted' : (0, (3, 1, 1, 1, 1, 1))
}


"""2-site example for R1rho with variable w1 field which is stepped in sync with the rf position"""
#%%
def palmerfigures(rows,cols,choice,x,y,col='black',ls='solid',newfig=True,newax=True):
    if newfig == True:
        fig, ax = plt.subplots(rows, cols, 
                        figsize=(10/2.54,14/2.54))
        ax[choice].clear()
        #plt.clf()
    else:
        #ax=[0 for i in np.arange(rows*cols)]
        fig,ax=newfig
        if newax == True:
            ax[choice].clear()
        #ax[choice]=axi
    #fig.subplots_adjust(hspace=0.15,wspace=0.1,top=1.0,bottom=0.0)
    fig.tight_layout()
  #  print(x)
    if x != []:
        ax[choice].plot(x,y,color=col,ls=ls)
    return fig,ax

fig1, ax1 = palmerfigures(2,1,1,[],[],newfig=True)
#palmerfigures(2,1,0,x,y,newfig=[fig1, ax1],col=orange,ls='dotted',newax=True)



#%%

def BMCcartesian(omegaA,R1A,R2A,omegaB,R1B,R2B,omega1,k1,km1):
    LA = np.array([
        [-R2A - k1,-omegaA,0],
        [omegaA, -R2A-k1,-omega1],
        [0,omega1,-R1A-k1]
    ])
    
    LB = np.array([
        [-R2B - km1,-omegaB,0],
        [omegaB, -R2B-km1,-omega1],
        [0,omega1,-R1B-km1]
    ])
    
    k1I = np.eye(3)*k1
    km1I = np.eye(3)*km1
    L = np.block([[LA,km1I],[k1I,LB]])
    return L

def calcfig5(Om,P,kex,R1,R2,rex=1,alt=2):
    """calculates data for Fig. 5a and related. 2-site, expandable
    Om=list of omA,omB
    P=list of pA,pB
    kex=k12+k21
    R1=list of R1s
    R2=list of R2s: R2A,R2B 
    alt: concerns the definition of omega_A and omega_B.
           True: omega_A=-w1a; omega_B=-w1a+dw
           False: omega_A=-pB*dw-w1a; omega_B=-pB*dw-w1a+dw
    """
    numbv=5000
    pA,pB=P
    omA,omB=Om
#    k12=RDmath.kv('12',[kex,0,0,0,0,0,pB,0,0])
    dw = np.abs(omB-omA)
    if alt == 2:
        w1a = dw/2
    else:
        w1a = np.linspace(1,2*dw,numbv)
    rf0 = np.linspace(0.1,2*dw,5000)   #####X


    if rex == 2:
        R1r0 = np.zeros(len(rf0))
        for i,wrf0 in enumerate(rf0):
            temp = BMCcartesian(-wrf0,R1[0],R2[0],dw-wrf0,R1[1],R2[1],w1a,0,0)
            tempeig = eig(temp)[0]
            tempeigR = tempeig[np.abs(np.imag(tempeig)) < 1e-6]
            R1r0[i] = np.min(np.abs(tempeigR))
            if wrf0 < dw/2:
                R1r0[i] = np.max(np.abs(tempeigR))

    if alt == 0:
        omega_A=-w1a
        omega_B=-w1a+dw
    elif alt == 1:                         
        omega_A=-pB*dw-w1a
        omega_B=-pB*dw-w1a+dw
    elif alt == 2:
        omega_A=-np.linspace(1,2*dw,numbv)
        omega_B=-np.linspace(1,2*dw,numbv)+dw
        #omega_A=-w1a
        #omega_B=-w1a+dw
    ombar = pA*omega_A + pB*omega_B#-rfa
    if alt == 2:
        res=[RDmath.rexeq(-i,dw,pB,kex,2,w1a,0,0,0,0,0,0,0,0,0,0,0,r1=R1,r2=R2) for j,i in enumerate(omega_A)]
    else:
        res=[RDmath.rexeq(-i,dw,pB,kex,2,w1a[j],0,0,0,0,0,0,0,0,0,0,0,r1=R1,r2=R2) for j,i in enumerate(omega_A)]
    we = np.sqrt(ombar**2 + w1a**2)
    sinthet = w1a/we
    costhet = ombar/we
    y=res*(sinthet**2)

    if rex == 2:
        y=(y-R1r0)/sinthet**2
    elif rex == 1:
        y=RDmath.Rex_n(y,sinthet,costhet,P,R1,R2)

    if alt == 2:
        return (omega_A)/(2*np.pi), y
    else:
        return (w1a)/(2*np.pi), y

omB = 2*np.pi*200
pA = 0.9
kexfac=1#2.5#0.1


#x,y=calcfig5b([0,omB],[pA,1-pA],omB*0.25,[1,1],[20,20])
#fig, ax = palmerfigures(2,1,1,-x,y,newfig=[fig1, ax1],col=blue,ls='dashed',newax=True)
kexfac=1
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=1,rex=0)
fig, ax = palmerfigures(2,1,0,x,y,newfig=[fig1, ax1],col=orange,ls='dotted',newax=True)
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=0,rex=0)
fig, ax = palmerfigures(2,1,0,x,y,newfig=[fig1, ax1],col=green,newax=False)
kexfac=2.5
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=1,rex=0)
fig, ax = palmerfigures(2,1,0,x,y,newfig=[fig1, ax1],col=reddishpurple,newax=False,ls=linestyles_dict['dashdotted2'])
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=0,rex=0)
fig, ax = palmerfigures(2,1,0,x,y,newfig=[fig1, ax1],col=blue,newax=False,ls=linestyles_dict['dashed'])

kexfac=1
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=2,rex=2)
fig, ax = palmerfigures(2,1,1,-x,y,newfig=[fig1, ax1],col=green,ls='solid',newax=True)
kexfac=2.5
x,y=calcfig5([0,omB],[pA,1-pA],omB*0.1*kexfac,[1,1],[20,20],alt=2,rex=2)
fig, ax = palmerfigures(2,1,1,-x,y,newfig=[fig1, ax1],col=blue,ls='dashed',newax=False)

ax[1].set_ylim([0,90])
plt.draw()

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

def getmidexp_2(omA,omB,omC,k12ex,k13ex,k23ex,pB,pC,w10,rf0,dw,dwc,R1,R2):
    """N-site middle experiment work-in-progrerss"""
    ombar0 = pA*omA + pB*omB+ pC*omC
    tanthet0 = w10/(ombar0-rf0)
    
    #that could be converted to an input parameter of some kind but maybe more convenient for now
    firstrf=1;resolution=200
    rfa = np.linspace(firstrf,2*np.max([omB,omC]),resolution)
    
    ombar = pA*(omA-rfa-ombar0) + pB*(omB-rfa-ombar0)+ pC*(omC-rfa-ombar0)
    w1a = np.abs(tanthet0*(ombar))
    we = np.sqrt(ombar**2 + w1a**2)
    sinthet = w1a/we
    costhet = ombar/we
    rexaltc=np.array([RDmath.rexeq(i+ombar0,dw+i+ombar0,pB,k12ex,2,w1a[j],0,0,dwc+i+ombar0,pC,k13ex,k23ex,0,0,0,0,0) for j,i in enumerate(rfa)])
    x=(w1a)/(2*np.pi)
    y=RDmath.R1r_n(rexaltc,sinthet,costhet,[pA,pB,pC],R1,R2)
    return x,y

#fig, ax = palmerfigures(2,1,1,[1,3,3],[3,4,5],newfig=[fig,ax],col=reddishpurple)


fig, ax = palmerfigures(2,1,1,[1,3,3],[3,4,5],newfig=[fig,ax],col=reddishpurple)

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

