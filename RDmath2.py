#!/usr/bin/env python3
"""
@author: hanskoss

mathematical equations for calculating R1r, CEST, and CPMG experiments
currently upgrading to python3
note to self: some small bug in CEST equation here, not relevant for paper.
adding equations for HAM-based R1rho calculations

"""
from __future__ import division
from hkimports2 import smp
from hkimports2 import np
from hkimports2 import scp
from hkimports2 import scplin
from scipy.linalg import expm

#from hkimports2 import expm


def Laguerre(omegaA,R1A,R2A,omegaB,R1B,R2B,omega1,k1,km1,exptype='R1r'):
    """2-site Laguerre approcimation, can obtain Rex or R1r"""
    
    kex = k1 + km1
    pA = 1.0
    pB = 0.0
    if kex > 1e-6: 
        pA = km1/kex
        pB = 1.0 - pA
    
    R1 = pA*R1A + pB*R1B
    R2 = pA*R2A + pB*R2B
    
    wA = np.sqrt(omegaA**2 + omega1**2)
    wB = np.sqrt(omegaB**2 + omega1**2)
    
    ombar = pA*omegaA + pB*omegaB
    we = np.sqrt(ombar**2 + omega1**2)
    sinthet = omega1/we
    costhet = ombar/we
    
    R1r = sinthet**2*R2 + costhet**2*R1
    
    Rexnum = pA*pB*sinthet**2*(omegaB-omegaA)**2
    
    temp1 = 1+2*kex**2*(pA*wA**2 + pB*wB**2)/(wA**2*wB**2 + we**2*kex**2)
    
    Rex = Rexnum*kex/(wA**2*wB**2/we**2 + kex**2 - Rexnum*temp1)
    if exptype == 'R1r':
        return Rex + R1r
    else:
        return Rex/sinthet**2
    
"""converting Rex to R1r and vice versa, works for n-site"""

def Rex_n(r1r,sinthet,costhet,pN,R1N,R2N):
    R2sum=np.sum([pN[j]*R2N[j] for j in np.arange(len(pN))])
    R1sum=np.sum([pN[j]*R1N[j] for j in np.arange(len(pN))])
    R1r0 = sinthet**2*R2sum + costhet**2*R1sum
    return (r1r-R1r0)/sinthet**2

def R1r_n(rex,sinthet,costhet,pN,R1N,R2N):
    R2sum=np.sum([pN[j]*R2N[j] for j in np.arange(len(pN))])
    R1sum=np.sum([pN[j]*R1N[j] for j in np.arange(len(pN))])
    R1r0 = sinthet**2*R2sum + costhet**2*R1sum
    #print(R1r0,'R1r0', sinthet**2,R2sum, costhet**2,R1sum)
    return rex*(sinthet**2)+R1r0


smp.init_printing(pretty_print=True,num_columns=150)
def anyisnotsymbol(listx,eq0):
    """This tests whether any element in a given list (listx)
    is not a symbol while also being greater 0. if eq0 is set to 1, 
    if any element in a given list is 0, but not a symbol. if eq0 is set to 2, 
    true is returned if none of the elements in the list is a symbol but any
    element is greater 0
    """
    return sum([((x>0)==1) for x in listx])>0 if eq0 == 0 else (sum([(x==0) \
    for x in listx])>0 if eq0 == 1 else sum([((x>0)==1) for x in listx])>0 \
    if eq0 == 0 else (sum([(x>0)!=1 and (x>0)!=0 for x in listx])+\
    sum([(sum([((x>0)==1 or (x>0)==0) and x > 0 for x in listx])>0)==0]))==0)
    #return 

def anyissymbol(listx):
    """This tests whether any element in a given list (listx) is a symbol
    """
  # print(listx)
  #  print(sum([((x>0)!=1 and (x>0)!=0) for x in listx])>0), 'symbtest'
    return sum([((x>0)!=1 and (x>0)!=0) for x in listx])>0

def kv(kvtype,parlist):
    [k12,k13,k23,k14,k24,k34,pb,pcx,pdx]=parlist
    kvtypelist=('12','21','13','31','23','32','14','41','24','42','34','43')
    #pc=pcx if ((anyisnotl([k13,k23],3) or anyissymbol([k13,k23]))) else 0
    [pc,pd]=[pcx,pdx] if ((anyisnotsymbol([k14,k24,k34],3) or \
            anyissymbol([k14,k24,k34]))) else ([pcx,0] if ((anyisnotsymbol(\
                       [k13,k23],3) or anyissymbol([k13,k23]))) else [0,0])
    pa=1-pb-pc-pd
#    print(anyissymbol([pd]), pd)
    if anyissymbol([pd]) or pd > 0:
        expressionlist=(pb*k12/(1-pc-pd),pa*k12/(1-pc-pd),pc*k13/(1-pb-pd),\
                        pa*k13/(1-pc-pd),pc*k23/(pb+pc),pb*k23/(pb+pc),pd*k14/\
                        (1-pb-pc),pa*k14/(1-pc-pc),pd*k24/(pb+pd),pb*k24/\
                        (pb+pd),pd*k34/(pc+pd),pc*k34/(pc+pd))
    elif anyissymbol([pc]) or pc > 0:
        expressionlist=(pb*k12/(1-pc),pa*k12/(1-pc),pc*k13/(1-pb),pa*k13/\
                        (1-pc),pc*k23/(pb+pc),pb*k23/(pb+pc))
    else:
        expressionlist=(pb*k12,pa*k12)
    return expressionlist[kvtypelist.index(kvtype)]



def approxdefs(**kwargs):
    """general function to calculate a number of expressions used to get
    approximations and exact solutions.
    This function frequently calls itself.
    Pre-calculated symbolic solutions are also generated from expressions
    contained herein.
    However, the function most likely directly used by the user will be
    nEVapprox.
    The parameter "calctype" determines which type of calculation is performed.
    """
    verb='quiet'


    #Parse arguments (parameters). Strings in the symbols list (sparlist)
    #will either take a value or become a symbol within the prm dictionary.
    #Strings in the other list (oparlist) will either take the value (from
    #kwargs or be set to 0:
    sparlist=['pa','pb','pc','pd','t','tt','k12','k13','k14','k23','k24','k34'\
              ,'dwa','dwb','dwc','dwd','w1','deltao']
    oparlist=['calctype','exporder','mode','pade','sc','nDV','nhmatnewsc']
    lparlist=['r1','r2']
    prm={}
    for x in sparlist+oparlist+lparlist:
        if x in kwargs:
            prm[x]=kwargs[x]
        elif x in sparlist:
            prm[x]=smp.Symbol(x)
        elif x in lparlist:
            prm[x]=[0,0,0,0]
        else:
            prm[x]=0
    #We prefer variables (corresponding to values or symbols)
    #over bulky dictionary entries, which is why all dictionary entries are
    #assigned to corresponding values:
    pa=prm['pa'];pb=prm['pb'];pc=prm['pc'];pd=prm['pd'];t=prm['t'];tt=prm\
        ['tt'];k12=prm['k12'];k13=prm['k13'];k14=prm['k14'];k23=prm['k23'];\
        k24=prm['k24'];k34=prm['k34'];dwb=prm['dwb'];dwc=prm['dwc'];dwd=prm\
        ['dwd'];calctype=prm['calctype'];exporder=prm['exporder'];mode=prm\
        ['mode'];pade=prm['pade'];sc=prm['sc'];nDV=prm['nDV'];dwa=prm['dwa'];\
        w1=prm['w1'];deltao=prm['deltao']
    r1=prm['r1']
    r2=prm['r2']
    #print(r1,'r1','!!!!!!',calctype)
 #   print(dwa,'dwahere',calctype)
    #If verb is set to 'ose', then the calculation type is printed.
    if verb == 'ose':
        print('calc '+calctype+'... ')
    
    #calculation of the chemical shift matrix - including the exponential:

    #calculation of the chemical shift matrix - not including the
    #exponential. Similar to nAmatD. Only works in numerical mode.
    elif calctype == 'nAmatDnonex':
        if mode ==2:
            return np.matrix([[0,0,0,0],[0,-1j*dwb,0,0],[0,0,-1j*dwc,0],\
                [0,0,0,-1j*dwd]]) if anyisnotsymbol([k14,k24,k34],0) else \
                (np.matrix([[0,0,0],[0,-1j*dwb,0],[0,0,-1j*dwc]]) if \
                anyisnotsymbol([k23,k13],0) else np.matrix([[0,0],\
                [0,-1j*dwb]]))
    

    elif calctype == 'nAmatP':
        if mode == 1:
            [k13,k23,k14,k24,k34]=[k13,0,0,0,k34] if sc == 0 else \
            ([k13,0,k14,0,0] if sc == 1 else ([k13,0,k14,0,k34] if sc == 2 \
            else ([k13,0,0,k24,k34] if sc == 3 else ([0,k23,0,0,0] if sc == 4 \
            else ([k13,0,0,0,0] if sc == 5 else ([k13,k23,0,0,0] if sc == 6 \
            else [0,0,0,0,0]))))))
        pl=[k12,k13,k23,k14,k24,k34,pb,pc,pd]
        if ((anyisnotsymbol([k14,k24,k34],3) or anyissymbol([k14,k24,k34]))):
            kineticmat=[[-kv('12',pl)-kv('13',pl)-kv('14',pl),kv('21',pl),\
            kv('31',pl),kv('41',pl)],[kv('12',pl),-kv('21',pl)-kv('23',pl)\
            -kv('24',pl),kv('32',pl),kv('42',pl)],[kv('13',pl),kv('23',pl),\
            -kv('31',pl)-kv('32',pl)-kv('34',pl),kv('43',pl)],[kv('14',pl),\
            kv('24',pl),kv('34',pl),-kv('41',pl)-kv('42',pl)-kv('43',pl)]]
        elif ((anyisnotsymbol([k13,k23],3) or anyissymbol([k13,k23]))):
            kineticmat=[[-kv('12',pl)-kv('13',pl),kv('21',pl),kv('31',pl)],\
            [kv('12',pl),-kv('21',pl)-kv('23',pl),kv('32',pl)],[kv('13',pl),\
            kv('23',pl),-kv('31',pl)-kv('32',pl)]]
        else:
            kineticmat=[[-kv('12',pl),kv('21',pl)],[kv('12',pl),-kv('21',pl)]]
        if mode == 2:
            return np.matrix(kineticmat)
        else:
            return smp.Matrix(kineticmat)
    

    ##approxdefs(calctype='r1rLKmat',mode=mode,k12=k12x,k13=k13x,k14=k14x,k24=k24x,k23=k23x,k34=k34x,pb=pbx,pc=pcx,pd=pdx,deltao=deltaox,w1=w1x,dwb=dwbx,dwc=dwcx,dwd=dwdx)
    
    #elif calctype == 'cpmg1':
    #    return np.dot(np.dot(scplin.expm(approxdefs(t=t,dwb=dwb,calctype='nAmat',mode=2)*t),scplin.expm(approxdefs(t=t,dwb=dwb,calctype='nAstar')*t*2)),scplin.expm(approxdefs(t=t,dwb=dwb,calctype='nAmat')*t))
    
    #calculates the exact solution for the exponential matrix term
    elif calctype == 'cpmg2':
        part1=approxdefs(calctype='nAmatP',mode=2,k12=k12,k13=k13,k14=k14,k24=\
               k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd)
        part2=approxdefs(calctype='nAmatDnonex',mode=2,k12=k12,k13=k13,k14=k14\
               ,k24=k24,k23=k23,k34=k34,dwb=dwb,dwc=dwc,dwd=dwd)
        return np.dot(np.dot(scplin.expm((part1+part2)*t),\
                             scplin.expm((part1-part2)*2*t)),\
                             scplin.expm((part1+part2)*t))
    #elif calctype == 'cpmg2old':
    #    return np.dot(scplin.expm((approxdefs(calctype='nAmatP',mode=2)+approxdefs(calctype='nAmatDnonex',mode=2))*2*t),scplin.expm((approxdefs(calctype='nAmatP',mode=2)-approxdefs(calctype='nAmatDnonex',mode=2))*2*t))
    

    #For calculations of Hmat
    elif calctype == 'nHmatpro':
        #calculations which are based on calculations of the
        #exact exponential matrix. (all ExpExact)
        if mode == 2 and (exporder == 0 or exporder == 10):
            cpmg2mat=approxdefs(mode=2,calctype='cpmg2',t=t,\
                       k12=k12,k13=k13,k14=k14,k24=k24,k23=k23,k34=k34,dwb=dwb\
                       ,dwc=dwc,dwd=dwd,pb=pb,pc=pc,pd=pd)
            if pade == 0: #exact logarithm
                return scplin.logm(cpmg2mat)/(t*4)
            elif pade == 1: #Pade 1,0
                return (-np.identity(np.shape(cpmg2mat)[0])+cpmg2mat)/(t*4)
            if pade == 2 or pade ==3:
                xmt=-approxdefs(mode=2,pade=1,calctype='nHmatpro',t=t,\
                        exporder=exporder,k12=k12,k13=k13,k14=k14,k24=k24,\
                        k23=k23,k34=k34,dwb=dwb,dwc=dwc,dwd=dwd,pb=pb,pc=pc,\
                        pd=pd)*(4*t)
            if pade == 2: #Pade 2,2
                return -(np.dot(np.linalg.inv(6*np.identity(np.shape(xmt)\
                        [0])-6*xmt+np.dot(xmt,xmt)),(6*xmt-np.dot(3*xmt,xmt\
                        ))))/(4*t)
            elif pade ==3: #Pade 3,3
                return -(np.dot(np.linalg.inv(60*np.identity(np.shape(xmt)[0])\
                        -90*xmt+np.dot(36*xmt,xmt)-np.dot(3*xmt,np.dot(xmt,\
                        xmt))),(60*xmt-np.dot(60*xmt,xmt)+np.dot(11*xmt,\
                        np.dot(xmt,xmt)))))/(4*t)

    elif calctype == 'r1rBigL2':
        dwlist=[dwa,dwb,dwc,dwd]
        Lmat=[[[-r2[xn],-dwlist[xn],0],[dwlist[xn],-r2[xn],-w1],[0,w1,-r1[xn]]] for xn,x in enumerate(r2)]
        idm=np.matrix(np.identity(3))
        if mode ==2:
            
            dimen = 4 if anyisnotsymbol([k14, k24, k34],0) else (3 if \
                anyisnotsymbol([k13,k23],0) else 2)
        else:
            dimen = 4 if sc < 4 else (3 if sc < 7 else 2)
        if dimen == 4:
            BigL=np.reshape(np.array([np.transpose(np.reshape(np.array(\
            [Lmat[0],0*idm,0*idm,0*idm]),(12,3))),np.transpose(np.reshape(\
            np.array([0*idm,Lmat[1],0*idm,0*idm]),(12,3))),np.transpose(\
            np.reshape(np.array([0*idm,0*idm,Lmat[2],0*idm]),(12,3))),np.\
            transpose(np.reshape(np.array([0*idm,0*idm,0*idm,Lmat[3]]),\
            (12,3)))]),(12,12),order='A')
        elif dimen == 3:
            BigL=np.reshape(np.array([np.transpose(np.reshape(np.array(\
            [Lmat[0],0*idm,0*idm]),(9,3))),np.transpose(np.reshape(\
            np.array([0*idm,Lmat[1],0*idm]),(9,3))),np.transpose(\
            np.reshape(np.array([0*idm,0*idm,Lmat[2]]),(9,3)))]),\
            (9,9),order='A')
        else:
            BigL=np.reshape(np.array([np.transpose(np.reshape(np.array(\
            [Lmat[0],0*idm]),(6,3))),np.transpose(np.reshape(\
            np.array([0*idm,Lmat[1]]),(6,3)))]),\
            (6,6),order='A')
        if mode == 2:
            return np.matrix(BigL)
        else:
            return smp.Matrix(BigL)
        
    elif calctype == 'nAmatP2':
        if mode == 1:
            [k13,k23,k14,k24,k34]=[k13,0,0,0,k34] if sc == 0 else \
            ([k13,0,k14,0,0] if sc == 1 else ([k13,0,k14,0,k34] if sc == 2 \
            else ([k13,0,0,k24,k34] if sc == 3 else ([0,k23,0,0,0] if sc == 4 \
            else ([k13,0,0,0,0] if sc == 5 else ([k13,k23,0,0,0] if sc == 6 \
            else [0,0,0,0,0]))))))
        pl=[k12,k13,k23,k14,k24,k34,pb,pc,pd]
        if ((anyisnotsymbol([k14,k24,k34],3) or anyissymbol([k14,k24,k34]))):
            kineticmat=[[-kv('12',pl)-kv('13',pl)-kv('14',pl),kv('21',pl),\
            kv('31',pl),kv('41',pl)],[kv('12',pl),-kv('21',pl)-kv('23',pl)\
            -kv('24',pl),kv('32',pl),kv('42',pl)],[kv('13',pl),kv('23',pl),\
            -kv('31',pl)-kv('32',pl)-kv('34',pl),kv('43',pl)],[kv('14',pl),\
            kv('24',pl),kv('34',pl),-kv('41',pl)-kv('42',pl)-kv('43',pl)]]
        elif ((anyisnotsymbol([k13,k23],3) or anyissymbol([k13,k23]))):
            kineticmat=[[-kv('12',pl)-kv('13',pl),kv('21',pl),kv('31',pl)],\
            [kv('12',pl),-kv('21',pl)-kv('23',pl),kv('32',pl)],[kv('13',pl),\
            kv('23',pl),-kv('31',pl)-kv('32',pl)]]
        else:
            kineticmat=[[-kv('12',pl),kv('21',pl)],[kv('12',pl),-kv('21',pl)]]
        if mode == 2:
            return np.matrix(kineticmat)
        else:
            return smp.Matrix(kineticmat)

    elif calctype == 'r1rBigK2':
        ktest=np.array(approxdefs(calctype='nAmatP2',mode=mode,k12=k12,k13=k13,k14=k14,k24=\
               k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,sc=sc))
        if mode == 1:
            ident=smp.eye(3)
            z=smp.zeros(np.shape(ktest)[0]*3)
        else:
            ident=np.identity(3)
            z=np.zeros((np.shape(ktest)[0]*3,np.shape(ktest)[0]*3))
        i=0
        for x in ktest:
            j=0
            for y in x:
        #       print(y)
      #         print(ident)
                z[i:i+3,j:j+3]=y*ident
                j+=3
            i+=3
        if mode == 2:
            return np.matrix(z)
        else:
            return smp.Matrix(z)
    
    elif calctype == 'r1rLKmat2':
 #       print(approxdefs(calctype='r1rBigK2',mode=mode,k12=k12,k13=k13,k14=k14\
#            ,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,sc=sc),'bigk')
  #      print(approxdefs(\
     #   calctype='r1rBigL2',mode=mode,sc=sc,k12=k12,k13=k13,k14=k14,k24=\
     #   k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,dwa=-(deltao-pb*dwb-pc*dwc-\
     #   pd*dwd),dwb=dwb-(deltao-pb*dwb-pc*dwc-pd*dwd),dwc=dwc-(deltao-\
     #   pb*dwb-pc*dwc-pd*dwd),dwd=dwd-(deltao-pb*dwb-pc*dwc-pd*dwd),w1=w1),'bigl')
#        return approxdefs(calctype='r1rBigK2',mode=mode,k12=k12,k13=k13,k14=k14\
#            ,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,sc=sc)+approxdefs(\
#            calctype='r1rBigL2',mode=mode,sc=sc,k12=k12,k13=k13,k14=k14,k24=\
#            k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,dwa=-(deltao-pb*dwb-pc*dwc-\
#            pd*dwd),dwb=dwb-(deltao-pb*dwb-pc*dwc-pd*dwd),dwc=dwc-(deltao-\
#            pb*dwb-pc*dwc-pd*dwd),dwd=dwd-(deltao-pb*dwb-pc*dwc-pd*dwd),w1=w1)
        return approxdefs(calctype='r1rBigK2',mode=mode,k12=k12,k13=k13,k14=k14\
            ,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,sc=sc)+approxdefs(\
            calctype='r1rBigL2',mode=mode,sc=sc,k12=k12,k13=k13,k14=k14,k24=\
            k24,k23=k23,k34=k34,pb=pb,pc=pc,pd=pd,dwa=-deltao,dwb=dwb-deltao,dwc=dwc-deltao,dwd=dwd-deltao,w1=w1,r1=r1,r2=r2)

                                                                   
    elif calctype == 'sinsqth2':
        if sc < 4:
            return (w1**2/(w1**2+((1-pb-pc-pd)*dwa+pb*dwb+pc*dwc+pd*dwd)**2))
        elif sc < 7:
            return (w1**2/(w1**2+((1-pb-pc)*dwa+pb*dwb+pc*dwc)**2))
        else:
  #          print(dwa,dwb,'dwadwb')
            return (w1**2/(w1**2+((1-pb)*dwa+pb*dwb)**2))
    elif calctype == 'r1rex':
# There were some minor errors in here. This is commented out for the time being. Remove in later versions.        
        #print(deltao,pb,dwb,-(deltao-pb*dwb-pc*dwc-pd*dwd),'r1rconv')
#        LKM=approxdefs(calctype='r1rLKmat2',mode=mode,k12=k12,k13=k13,\
#           k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
#           pd=pd,deltao=deltao,w1=w1,dwa=-(deltao-pb*dwb-pc*dwc-\
#            pd*dwd),dwb=dwb-(deltao-pb*dwb-pc*dwc-pd*dwd),dwc=dwc-(deltao-\
#            pb*dwb-pc*dwc-pd*dwd),dwd=dwd-(deltao-pb*dwb-pc*dwc-pd*dwd),sc=sc)
#        LKM=approxdefs(calctype='r1rLKmat2',mode=mode,k12=k12,k13=k13,\
#           k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
#           pd=pd,deltao=deltao,w1=w1,dwa=0-pb*dwb-pc*dwc-pd*dwd,dwb=dwb-pb*dwb-pc*dwc-pd*dwd,dwc=dwc-pb*dwb-pc*dwc-pd*dwd,dwd=dwd-pb*dwb-pc*dwc-pd*dwd,sc=sc)

        LKM=approxdefs(calctype='r1rLKmat2',mode=mode,k12=k12,k13=k13,\
            k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
            pd=pd,deltao=deltao,w1=w1,dwa=0,dwb=dwb,dwc=dwc,dwd=dwd,sc=sc,r1=r1,r2=r2)

   #     print(LKM, 'LKM')
#        sinsqt=approxdefs(calctype='sinsqth2',mode=mode,k12=k12,k13=k13,\
#           k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
#           pd=pd,deltao=deltao,w1=w1,dwa=-(deltao-pb*dwb-pc*dwc-\
#            pd*dwd),dwb=dwb-(deltao-pb*dwb-pc*dwc-pd*dwd),dwc=dwc-(deltao-\
#            pb*dwb-pc*dwc-pd*dwd),dwd=dwd-(deltao-pb*dwb-pc*dwc-pd*dwd),sc=sc)
        sinsqt=approxdefs(calctype='sinsqth2',mode=mode,k12=k12,k13=k13,\
           k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
           pd=pd,deltao=deltao,w1=w1,dwa=-(deltao),dwb=dwb-(deltao),dwc=dwc-(deltao),dwd=dwd-(deltao),sc=sc)
        if mode == 2:
   #         print(np.real(1/np.max(-1/np.linalg.eigvals(LKM))).item(),'xxxinloop')
            return np.real(1/np.max(-1/np.linalg.eigvals(LKM))).item()/sinsqt

    
   
def nEVapprox(*args,**kwargs):
    """Main function to generate exact solutions and approximations, except
    Carver-Richards.
    These arguments have to be supplied:
    t,dwb,pb,k12,mode,exporder,pade,evap,dwc,pc,k13,k23,dwd,pd,k14,k24,k34.
    t - tau; dwb, dwc, dwd - shifts of minor state peaks with respect to main
    peak, in s-1; pb, pc, pd: minor site populations; k12, k13, k23, k14, k24,
    k34 - exchange rate constants. For example, "k12" is actually corresponding 
    to k12+k21. in s-1. mode=1: returns symbolic solution if available. mode=2:
    return numerical solution (sometimes based on equations which have been
    precalculated symbolically in part).
    exporder: 0: exact solution for matrix exponential. 1-4: higher order
    approximations for slow exchange (only order 1 turns out to be useful).
    log: 0: exact solution for logarithm. 1: Pade(1,0); 2: Pade(2,2); 3: 
    Pade(3,3)
    evap: eigenvalue approximations:
    0: exact solution.
    1/1001/2001: First order solution.
    2,3,... Newton-Raphson
    1002,1003,... Laguerre approximation (not implemented in this version)
    2002,2003,.... Halley's method (only numerical)
    """
    [t,dwb,pb,k12,mode,exporder,pade,evap,dwc,pc,k13,k23,dwd,pd,k14,k24,k34]=\
        args
        
    #The following section selects the right H matrix. This it, at this time,
    #a little convoluted because parts of this decision tree are located in
    #the function approxdefs.
    #switches: exporder=0 and exporder=10 (right now) identical, ExpExact
    #exporder=N corresponding to ExpN approx for slow exchange
    #mode=1 - symbolic; mode=2 - numerical; pade=0: LogExact; pade=1: Log10
    #pade=2: Log22; pade=3: Log33.
    if exporder == 10 or exporder == 0 or pade == 0:
        mtx=approxdefs(mode=2,calctype='nHmatpro',exporder=exporder,pade=pade,\
        pb=pb,pc=pc,pd=pd,k12=k12,k13=k13,k14=k14,k23=\
        k23,k24=k24,k34=k34,dwb=dwb,dwc=dwc,dwd=dwd,t=t)
    
    #This selects and calculates the eigenvalue approximations
    #0: exact solution.
    #1/1001/2001: First order solution.
    #2,3,... Newton-Raphson
    #1002,1003,... Laguerre approximation (not implemented in this version)
    #2002,2003,.... Halley's method (only numerical)
    if evap == 1 or evap == 1001 or evap == 2001:
        if mode == 2:
            return -np.real(1/np.trace(np.linalg.inv(mtx))).item()
        else:
            mtg=smp.simplify(mtx)
            return -1/trace(mtg.inv())
    elif evap > 1001 and evap < 2000: #Laguerre higher order, not implemented
        if mode ==2:
            substi=-nEVapprox(t,dwb,pb,k12,2,exporder,pade,evap-1,dwc,pc,k13,\
                              k23,dwd,pd,k14,k24,k34)
            zlaguerre=np.linalg.inv(mtx-substi*np.identity(np.shape(mtx)[0]))
            return 0 ##### NOT IMPLEMENTED HERE, AND NOT USED IN PLOTS
        else:
            mtg=smp.simplify(mtx)
            substi = -smp.simplify(nEVapprox(t,dwb,pb,k12,1,exporder,pade,evap\
                    -1,dwc,pc,k13,k23,dwd,pd,k14,k24,k34))
            zlaguerre=smp.simplify((mtg-substi*smp.eye(np.shape(mtx)[0])).inv\
                    ())
            return 0 ##### NOT IMPLEMENTED HERE, AND NOT USED IN PLOTS
    elif evap > 2001: ##### Halley's method higher order
        if mode ==2: #numerical
            substi=-nEVapprox(t,dwb,pb,k12,2,exporder,pade,evap-1,dwc,pc,k13,\
                    k23,dwd,pd,k14,k24,k34)
            zlaguerre=np.linalg.inv(mtx-substi*np.identity(np.shape(mtx)[0]))
            return -np.real(substi+((2*np.trace(zlaguerre))/((np.trace(\
                    zlaguerre**2))+(np.trace(zlaguerre)**2)))).item()
        else: #symbolic (not implemented)
            mtg=smp.simplify(mtx)
            substi = -smp.simplify(nEVapprox(t,dwb,pb,k12,1,exporder,\
                    pade,evap-1,dwc,pc,k13,k23,dwd,pd,k14,k24,k34))
            zlaguerre=smp.simplify((mtg-substi*smp.eye(np.shape(mtx)[0])).inv\
                    ())
            return 0 #### NOT IMPLEMENTED HERE
    elif evap > 1: #Newton-Raphson higher order approximations
        if mode ==2: #numerical
            substi=-nEVapprox(t,dwb,pb,k12,2,exporder,pade,evap-1,dwc,pc,k13,\
                    k23,dwd,pd,k14,k24,k34)
            return -np.real((substi)*(1-1/np.trace(np.linalg.inv(np.identity(\
                    np.shape(mtx)[0])-mtx/substi)))).item()
        else: #symbolic
            mtg=smp.simplify(mtx)
            substi = -smp.simplify(nEVapprox(t,dwb,pb,k12,1,exporder,pade,evap\
                    -1,dwc,pc,k13,k23,dwd,pd,k14,k24,k34))
            #OTHER VERSION
            return -(substi)*(1-1/trace((smp.eye(np.shape(mtx)[0])-mtg/substi)\
                    .inv()))
    elif evap == 0: #exact solution (for eigenvalue)
        return np.real(1/np.max(-1/np.linalg.eigvals(mtx))).item()


def threetriangfunctionx(dwa,dwb,dwc,k12,k13,k23,pb,pc,w1,R1,R2):
    """Three-site R1rho-type exchange for CEST fits including R1 relaxation"""
    return np.array([[0,0,0,0,0,0,0,0,0,0],[0,-R2 - k12*pb/(-pc + 1) - k13*pc/(-pb + 1),dwa,0,k12*(-pb - pc + 1)/(-pc + 1),0,0,k13*(-pb - pc + 1)/(-pc + 1),0,0],[0,-dwa,-R2 - k12*pb/(-pc + 1) - k13*pc/(-pb + 1),w1,0,k12*(-pb - pc + 1)/(-pc + 1),0,0,k13*(-pb - pc + 1)/(-pc + 1),0],[2*R1*(1-pb-pc),0,-w1,-R1 - k12*pb/(-pc + 1) - k13*pc/(-pb + 1),0,0,k12*(-pb - pc + 1)/(-pc + 1),0,0,k13*(-pb - pc + 1)/(-pc + 1)],[0,k12*pb/(-pc + 1),0,0,-R2 - k12*(-pb - pc + 1)/(-pc + 1) - k23*pc/(pb + pc),dwb,0,k23*pb/(pb + pc),0,0],[0,0,k12*pb/(-pc + 1),0,-dwb,-R2 - k12*(-pb - pc + 1)/(-pc + 1) - k23*pc/(pb + pc),w1,0,k23*pb/(pb + pc),0],[2*R1*pb,0,0,k12*pb/(-pc + 1),0,-w1,-R1 - k12*(-pb - pc + 1)/(-pc + 1) - k23*pc/(pb + pc),0,0,k23*pb/(pb + pc)],[0,k13*pc/(-pb + 1),0,0,k23*pc/(pb + pc),0,0,-R2 - k13*(-pb - pc + 1)/(-pc + 1) - k23*pb/(pb + pc),dwc,0],[0,0,k13*pc/(-pb + 1),0,0,k23*pc/(pb + pc),0,-dwc,-R2 - k13*(-pb - pc + 1)/(-pc + 1) - k23*pb/(pb + pc),w1],[2*R1*pc,0,0,k13*pc/(-pb + 1),0,0,k23*pc/(pb + pc),0,-w1,-R1 - k13*(-pb - pc + 1)/(-pc + 1) - k23*pb/(pb + pc)]])

def cestfunction(omegarflist,deltaAB,deltaAC,k12,k13,k23,pb,pc,w1x,R1,R2,B0,exactx):
    """
    CEST functions to fit data. Input variables are  mostly self-explanatory. exactx
    lets chose between exact solution and approximation. trad is, at the moment, hard-coded.
    The position of the CEST dip, which is important and refers to "position 0" depends
    on whether the dominant site is in slow or fast exchange with any of the minor site(s).
    This position is calculated initially.
    //// THIS MIGHT NOT BE THE MOST UPDATED VERSION. recall issues in collaborative project Feb 2022 with large minor site pop.
    
    """

    cest=[]
    #R1=10
    trad=0.4
    deltaA0=0
    w1=w1x*(2*np.pi)
    if k12 > deltaAB*B0 and k13 <= deltaAC*B0:
        deltaA0=-deltaAB*pb/(1-pc)
        deltaB0=deltaAB-deltaAB*pb/(1-pc)
        deltaC0=deltaAC-deltaAB*pb/(1-pc)
    elif k12 <= deltaAB*B0 and k13 <= deltaAC*B0:
        deltaA0=0
        deltaB0=deltaAB
        deltaC0=deltaAC
    elif k12 <= deltaAB*B0 and k13 > deltaAC*B0:
        deltaA0=-deltaAC*pc/(1-pb)
        deltaB0=deltaAB-deltaAC*pc/(1-pb)
        deltaC0=deltaAC-deltaAC*pc/(1-pb)
    elif k12 > deltaAB*B0 and k13 > deltaAC*B0:
        deltaA0=-deltaAB*pb-deltaAC*pc
        deltaB0=deltaAB-deltaAB*pb-deltaAC*pc
        deltaC0=deltaAC-deltaAB*pb-deltaAC*pc
    
    for omegarf in omegarflist:
        dwa = (deltaA0*B0-(2*np.pi)*omegarf*B0*500)
        dwb = (deltaB0*B0-(2*np.pi)*omegarf*B0*500)
        dwc = (deltaC0*B0-(2*np.pi)*omegarf*B0*500)
        omegaBar = (1-pb-pc)*dwa +pb*dwb + pc*dwc
        we = np.sqrt(w1**2 + omegaBar**2)
        cos2t = (omegaBar/we)**2
        
        if exactx == 1:
            Z=threetriangfunctionx(dwa,dwb,dwc,k12,k13,k23,pb,pc,w1,R1,R2)
            at =expm(trad*Z)    
            m0 = np.array([0.5,0,0,1-pb-pc,0,0,pb,0,0,pc])
            m1= np.array([0.5,0,0,-(1-pb-pc),0,0,-pb,0,0,-pc])
            magA=at[3,3]*m0[3]+at[3,6]*m0[6]+at[3,9]*m0[9]
            magA=magA-(at[3,3]*m1[3]+at[3,6]*m1[6]+at[3,9]*m1[9])
            magB=at[6,3]*m0[3]+at[6,6]*m0[6]+at[6,9]*m0[9]
            magB=magB-(at[6,3]*m1[3]+at[6,6]*m1[6]+at[6,9]*m1[9])
            magC=at[9,3]*m0[3]+at[9,6]*m0[6]+at[9,9]*m0[9]
            magC=magC-(at[9,3]*m1[3]+at[9,6]*m1[6]+at[9,9]*m1[9])
            mag=(magA+magB+magC)/2
            cest.append(mag)

    return cest[0]/(cos2t*np.exp(-R1*trad))
    
def rexeq(*args,**kwargs):
    """This does return Rex, /// function name recently revised
    Assumes that the exact solution for Rex is the greatest non-negative eigenvalue.
    """
    [t,dwb,pb,k12,mode,w1,pade,evap,dwc,pc,k13,k23,dwd,pd,k14,k24,k34]=\
        args
    if 'r1' in kwargs:
        r1=kwargs['r1']
        r2=kwargs['r2']
    schemecond=[sum([k14 != 0,k34 == 0,k24 != 0]),sum([k14 == 0, \
                k24 != 0, k34 != 0]),sum([k14 == 0, k34 == 0, k24 \
                != 0]),sum([k34 == 0,k24 == 0, k14 != 0]),sum([k23 \
                == 0, k13 !=0]),sum([k23 != 0, k13 ==0]),sum([k23 \
                == 0, k13 ==0]),sum([k23 != 0, k13 !=0])]
    sc=[ x for x,m in enumerate(schemecond) if m == 0][0]
    #print(' ')
    #print('donow',t,dwb)
    if 'r1' in kwargs:    
        return approxdefs(calctype='r1rex',mode=2,k12=k12,k13=k13,\
               k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
               pd=pd,deltao=t,w1=w1,dwa=0,dwb=dwb,dwc=dwc,dwd=dwd,sc=sc,r1=r1,r2=r2)
    else:
        return approxdefs(calctype='r1rex',mode=2,k12=k12,k13=k13,\
               k14=k14,k24=k24,k23=k23,k34=k34,pb=pb,pc=pc,\
               pd=pd,deltao=t,w1=w1,dwa=0,dwb=dwb,dwc=dwc,dwd=dwd,sc=sc)
    
    #r1req[t,dwb,pb,k12,mode,w1,pade,evap,dwc,pc,k13,k23,dwd,pd,k14,k24,k34]

#pb=0.016
#dwx=2.3*90*np.pi*2
#wx=3.500#*2*np.pi
#t=int(round(1/(int(40000)/1000000),0))#1/20
#t2=int(round(float(wx)*3628,0))

#kex=4000
#print nEVapprox(1/4*np.array(t),dwx,pb,3000,2,0,0,0,0,0,0,0,0,0,0,0,0)
#print nEVapprox(1/4*np.array(t2),10000,pb,kex,2,0,0,0,300,0.0001,100,0,0,0,0,0,0)
limittest=0
if limittest == 1:
    wx=3.5#3#.500
    tSL=30000
    tx=[1/int(round(1/(int(tSL)/2000000),0)),1/int(round(float(wx)*3628,0))]
    #tx=[0.00757576,0.0000196881]
    dwx=-3149
    pbx=0.08
    kexx=35000
    
    #10.634108040493764 10.6414710278 -972.8306565300018 [1.49253731e-02 7.87525595e-05] 0.015636672138346457 21993.581154509604 3260.9438716664763 
    dcx=0#-880.7275061160019 
    pcx=0#0.01636462632854531 
    k13x=0#665.4799970018257 
    k23x=0#1.9999999999999947
    
    rex1=r1req(0,dwx,pbx,kexx,2,wx*2*np.pi,0,0,0,0,0,0,0,0,0,0,0)
    cpmg1=nEVapprox(tx[1],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    cpmg2=nEVapprox(tx[0],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    viarex=cpmg2-rex1
    viacpmg=cpmg2-cpmg1#hkRDmath.nEVapprox(tx[0],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)-hkRDmath.nEVapprox(tx[1],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    
    dwx=-695
    rex1=r1req(0,dwx,pbx,kexx,2,wx*2*np.pi,0,0,0,0,0,0,0,0,0,0,0)
    cpmg1=nEVapprox(tx[1],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    cpmg2=nEVapprox(tx[0],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    viarex2=cpmg2-rex1
    viacpmg2=cpmg2-cpmg1#hkRDmath.nEVapprox(tx[0],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)-hkRDmath.nEVapprox(tx[1],dwx,pbx,kexx,2,0,0,0,dcx,pcx,k13x,k23x,0,0,0,0,0)
    
    print(viacpmg)#, viacpmg2

