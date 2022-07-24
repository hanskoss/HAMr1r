#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:35:33 2019
@author: hanskoss
All this file does is importing modules not created by myself.
This is supposed to help getting an overview about 
important dependencies and packages. Might include more details later.
#not enitrely complete though#
"""
import sympy as smp
import numpy as np
import scipy as scp
import scipy.linalg as scplin
from sympy import flatten
import scipy.optimize as optimize
import csv
import os
from datetime import datetime, date
import sys
import dill as pickle
from matplotlib.image import NonUniformImage
from scipy.fftpack import fft,fftfreq,fftshift
smp.init_printing(pretty_print=True,num_columns=150)
import matplotlib.pyplot as plt
import time
import multiprocessing
import copy
from datetime import datetime, date
#import hk_cpmg_symbolic1_3
import matplotlib.gridspec as gridspec 
from scipy.linalg import expm
from scipy.optimize import curve_fit

