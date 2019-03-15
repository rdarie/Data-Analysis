"""
The following functions recreate the nonlinear model of the electrode tissue
interface from:

Howell, Bryan, Sagar Naik, and Warren M. Grill.
"Influences of interpolation error, electrode geometry, and the electrode–tissue
interface on models of electric fields produced by deep brain stimulation."
IEEE transactions on Biomedical Engineering 61.2 (2014): 297-307.
"""
import math as m
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from lmfit import Minimizer, Parameters, report_fit

def doubleLayerCapacitancePerArea(V, c_H=45, k_1=31, k_2=1.2):
    #c_H uF/cm^2
    #k_1 uF/cm^2
    #k_2 V^-1
    if abs(V) > 10:
        c_dl = c_H
    else:
        c_D = k_1 * m.cosh(k_2 * V)
        c_dlInv = 1 / c_H + 1 / c_D
        c_dl = 1 / c_dlInv

    # c_dl in uF / cm ^ 2
    return c_dl

def faradaicResistancePerArea(V, J_0=1e-4, r_min=1e-1):
     #J_0 A/cm^2
    alpha = 1/2
    beta = 37.44 # V^-1

    if abs(V) > 10:
        r_f = r_min
    else:
        r_f = 1 / (J_0 * (alpha * beta * m.exp(-alpha * beta * V) +
            (1-alpha) * beta * m.exp((1-alpha) * beta * V) ) ) + r_min

    # r_f in Ohm * cm ^ 2
    return r_f

"""
S in units of mm^2
V in units of volts
"""
def doubleLayerCapacitance(S, V=1, c_dl=None, c_H=45, k_1=31, k_2=1.2):
    if c_dl is None:
        c_dl = doubleLayerCapacitancePerArea(V, c_H = c_H, k_1 = k_1, k_2 = k_2)
    return c_dl * (S / 100)

def faradaicResistance(S, V=1, r_f=None, J_0=1e-4, r_min=1e-1):
    if r_f is None:
        r_f = faradaicResistancePerArea(V, J_0 = J_0, r_min = r_min)
    return r_f / (S / 100)

"""
Formulas from:
Wei, Xuefeng F., and Warren M. Grill.
"Impedance characteristics of deep brain stimulation electrodes
in vitro and in vivo."
Journal of neural engineering 6.4 (2009): 046008.
R_s for transients was 133 in vitro and 533 in vivo
ETI = electrode tissue interface
"""

def ETIImpedance(S, F = 1e3, V = 1, R_f = None, C_dl = None,
    J_0 = 1e-4, r_min = 1e-1,
    c_H = 45, k_1 = 31, k_2 = 1.2,
    ):
    if R_f is None:
        R_f = faradaicResistance(S, V, J_0 = J_0, r_min = r_min)
    if C_dl is None:
        C_dl = doubleLayerCapacitance(S, V, c_H = c_H, k_1 = k_1, k_2 = k_2)
    omega = 2 * m.pi * F

    Z = R_f / complex(1, omega * R_f *C_dl)
    return Z

def ETITransient(S, I, PW, R_s, openCircuitV = 0, r_f = None, c_dl = None,
    J_0 = 1e-4, r_min = 1e-1,
    c_H = 45, k_1 = 31, k_2 = 1.2,
    step = 1e-6):
    #PW in sec, I in A, S in mm^2, R_s in ohms
    curV = 0
    V = pd.Series(0, index = I.index)
    R_f = pd.Series(0, index = I.index)
    C_dl = pd.Series(0, index = I.index)

    for curT, curI in I.items():
        #pdb.set_trace()
        curR_f = faradaicResistance(S, curV, r_f = r_f, J_0 = J_0, r_min = r_min) # ohms


        curC_dl = doubleLayerCapacitance(S, curV, c_dl = c_dl, c_H = c_H, k_1 = k_1, k_2 = k_2) * 1e-6 # uF to F conversion

        I_R = curV / curR_f
        I_C = curI - I_R
        dVdt = I_C / curC_dl

        curV += dVdt * step

        V.loc[curT] = curV + curI * R_s + openCircuitV
        R_f.loc[curT] = curR_f
        C_dl.loc[curT] = curC_dl

    return V, R_f, C_dl

def ETIPulseResponse(S, IAmp, PW = None, pulseShape = 'biphasic-symmetric',
    R_s = 150, openCircuitV = 0, r_f = None, c_dl = None, step = 1e-6,
    J_0 = 1e-4, r_min = 1e-1,
    c_H = 45, k_1 = 31, k_2 = 1.2,
    F = None, nPeriods = None):

    if pulseShape == 'biphasic-symmetric':
        grid = PW / step
        t = np.linspace(- 0.5 * PW, 2.5 * PW, 3 * grid) # t in sec
        I = pd.Series(0, index = t)
        I.loc[np.logical_and(t > 0 , t < PW)] = IAmp
        I.loc[np.logical_and(t > PW , t < 2 * PW)] = - IAmp
    elif pulseShape == 'monophasic':
        grid = PW / step
        t = np.linspace(- 0.5 * PW, 1.5 * PW, 2 * grid) # t in sec
        I = pd.Series(0, index = t)
        I.loc[np.logical_and(t > 0 , t < PW)] = IAmp
    elif pulseShape == 'sine':
        PW = nPeriods / F # seconds
        step = 1e-6 # seconds
        grid = PW / step
        t = np.linspace(- 0.5 * PW, 1.5 * PW, 2 * grid) # t in sec
        sineWave = np.tile(np.sin(np.linspace(-m.pi, m.pi, 1 / (F * step))), nPeriods)
        I = pd.Series(0, index = t)
        I.loc[np.logical_and(t > 0 , t < PW)] = sineWave * IAmp

    V, R_f, C_dl = ETITransient(S, I, PW, R_s, openCircuitV = openCircuitV,
        r_f = r_f, c_dl = c_dl,
        J_0 = J_0, r_min = r_min,
        c_H = c_H, k_1 = k_1, k_2 = k_2,
        step = step)
    return I, V, R_f, C_dl

def fitETIPulseResponseLinear(
    V, indexIntoModel, PW, step, theParameters, pulseShape = 'biphasic-symmetric',
    F = None, nPeriods = None, plotting = False):

    def fcn2min(params, voltages):
        S = params['S'].value # mm^2
        IAmp = params['IAmp'].value # A
        openCircuitV = params['openCircuitV'].value # V
        R_s = params['R_s'].value # ohms
        r_f = params['r_f'].value # ohms * cm^2
        c_dl = params['c_dl'].value # uF / cm^2

        upSample = 25

        I, modelV, R_f, C_dl = ETIPulseResponse(S, IAmp, PW = PW, pulseShape = pulseShape,
            R_s = R_s, openCircuitV = openCircuitV, r_f = r_f, c_dl = c_dl,
            step = step / upSample,
            F = F, nPeriods = nPeriods)
        modelV = modelV.iloc[::upSample]

        repModelV = np.repeat(modelV.iloc[indexIntoModel].values[np.newaxis,:],
            voltages.shape[0], axis = 0)

        if plotting:
            print(params)
            plt.plot(range(data.shape[1]), voltages, 'k-', label = 'Target')
            plt.plot(range(data.shape[1]), modelV.iloc[indexIntoModel], 'r.', label = 'Linear Model')
            plt.legend()
            plt.show()

        return (voltages-repModelV).flatten()

    minner = Minimizer(fcn2min, theParameters, fcn_args=((V,)))
    return minner.minimize()

def fitETIPulseResponseNonLinear(
    V, I, indexIntoModel, PW, step, theParameters, upSample = 25, pulseShape = 'biphasic-symmetric',
    F = None, nPeriods = None, plotting = False):

    def fcn2min(params, currents, voltages):
        S = params['S'].value # mm^2

        if currents is None:
            IAmp = [params['IAmp'].value] # A
            currents = np.full(voltages.shape[0], params['IAmp'].value)
        else:
            IAmp = np.unique(currents)

        openCircuitV = params['openCircuitV'].value # V
        R_s = params['R_s'].value # ohms

        J_0 = params['J_0'].value # A / cm^2
        r_min = params['r_min'].value # A / cm^2

        c_H = params['c_H'].value # uF / cm^2
        k_1 = params['k_1'].value # uF / cm^2
        k_2 = params['k_2'].value # uF / cm^2

        allModelV = np.zeros(voltages.shape, dtype = np.float32)

        for current in IAmp:
            I, modelV, R_f, C_dl = ETIPulseResponse(S, current, PW = PW, pulseShape = pulseShape,
                R_s = R_s, openCircuitV = openCircuitV,
                J_0 = J_0, r_min = r_min,
                c_H = c_H, k_1 = k_1, k_2 = k_2,
                step = step / upSample,
                F = F, nPeriods = nPeriods)
            modelV = modelV.iloc[::upSample]

            locMask = currents == current
            repModelV = np.repeat(modelV.iloc[indexIntoModel].values[np.newaxis,:],
                int(locMask.sum()), axis = 0)
            allModelV[locMask, :] = repModelV

            if plotting:
                print(params)
                plt.plot(range(voltages.shape[1]), voltages[np.where(locMask)[0][0],:], 'k-', label = 'Target')
                plt.plot(range(voltages.shape[1]), modelV.iloc[indexIntoModel], 'r.', label = 'Nonlinear Model')
                plt.legend()
                plt.show()

        return (voltages-allModelV).flatten()

    minner = Minimizer(fcn2min, theParameters, fcn_args=(I,V))
    return minner.minimize()

def ETIImpedanceFromTransient(S, IAmp, PW, R_s = 150, openCircuitV = 0, r_f = None, c_dl = None, step = 1e-6):
    I, V, R_f, C_dl = ETIPulseResponse(S, IAmp, PW, pulseShape = 'monophasic', R_s = R_s, openCircuitV = openCircuitV, r_f = r_f, c_dl = c_dl, step = step)
    return (V.max()-openCircuitV) / IAmp - R_s

if __name__ == "main":
    S = 1.79 # mm^2
    IAmp = .4e-3 # A
    PW = 80e-6 # sec
    R_s = 0 # ohms
    openCircuitV = 0 # Volts
    """
    F = 1e3 # Hz
    nPeriods = 20 # count of periods

    I, V, R_f, C_dl = eti.ETIPulseResponse(S, IAmp, F = F, nPeriods = nPeriods,
        pulseShape = 'sine', R_s = R_s, openCircuitV = openCircuitV,
        step = 1e-6)
    """
    I, V, R_f, C_dl = eti.ETIPulseResponse(S, IAmp, PW,
        pulseShape = 'biphasic-symmetric', R_s = R_s, openCircuitV = openCircuitV,
        step = 1e-6)

    cmap=plt.get_cmap('Set1')
    fig, ax = plt.subplots(2,1,sharex = 'col')
    line1, = ax[0].plot(V.index * 1e3, V, label = 'Interface Voltage (nonlinear model)', color = cmap(0))
    #plt.ylim(0.44, 0.47)
    #plt.xlim(-0.5e-3, 2.5e-3)
    ax[0].set_ylabel('Voltage (V)')
    ax[0].set_xlabel('Time (msec)')
    ax[0].set_title('Response of %4.2f mm^2 electrode to %4.2f mA pulses (%4.0f Ohm tissue Impedance)' % (S, IAmp * 1e3, R_s))
    ax[0].grid(True)
    plt.legend()
    ax2 = ax[0].twinx()
    line2, = ax2.plot(I.index * 1e3, I * 1e3, label = 'Current', color = cmap(1))
    ax2.set_ylabel('Current (mA)')
    plt.legend(handles = (line1, line2))

    line3, = ax[1].plot(C_dl.index * 1e3, C_dl * 1e6, label = 'Double layer capacitance', color = cmap(2))
    ax[1].set_ylabel('Capacitance (uF)')
    ax[1].set_xlabel('time (msec)')
    ax[1].grid(True)
    ax2 = ax[1].twinx()
    line4, = ax2.plot(R_f.index * 1e3, R_f, label = 'Faradaic Resistance', color = cmap(3))
    ax2.set_ylabel('Resistance (Ohms)')
    plt.legend(handles = (line3, line4))
    plt.show()

    print('The estimated impedance is %4.2f Ohms' % ((V.max() - openCircuitV) / IAmp - R_s))

    """
    Z = eti.ETIImpedance(S, F = 1e3, R_f = R_f.mean(), C_dl = C_dl.mean()) + R_s

    VppSteadyState = V.loc[V.index > 15e-3].max() - V.loc[V.index > 15e-3].min()
    Zeffective = VppSteadyState / (IAmp * 2)

    r_f = R_f.mean() * (S / 100)
    c_dl = C_dl.mean() / (S / 100)
    """
