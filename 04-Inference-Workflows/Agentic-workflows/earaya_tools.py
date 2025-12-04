from typing import Any, Dict, Literal
import os
from langchain_core.tools import tool

from ase.io import write as ase_write
from ase import Atoms

from numpy import loadtxt
import numpy as np


@tool
def loadascii(inputfile,column_x,column_y,skiprows=0) -> list[float]:
    """
    Read two columns x and y from inputfile and return them as a python list. 

    Parameters
    ----------
    inputfile : str
        The name of the input ascii file in the current directory.

    column_x : int
        The number of the column to be saved in the x array. First column of the file is 1.

    column_y : int
        The number of the column to be saved in the y array. First column of the file is 1.

    skiprows : int
        Number of rows in the ascii file that have no data (e.g., header) to skip before reading data into the x and y arrays.

    Returns
    -------
    list[float]
        Python list [x,y] containing the data from the ascii file.

    """
    #from numpy import loadtxt
    #import numpy as np
    spectrum = loadtxt(inputfile,dtype="float",skiprows=skiprows)
    x = spectrum[:,column_x-1]
    y = spectrum[:,column_y-1]
    x = np.array(x)
    y = np.array(y)
    return [x,y]

@tool
def getsubdata(x,y,xmin=None,xmax=None) -> list[float]:
    #get a subset of data from x and y based on x range.
    """
    Get a subset of the data from the x and y arrays.

    Parameters
    ----------
    x : list[float]
        Python array x containing the velocity channels of the spectrum. 

    y : list[float]
        Python array y containing the flux density or intensity values of the spectrum.

    xmin : float
        Minimum velocity value (x axis) of the domain to be selected for the subset of data.

    xmax : float
        Maximum velocity value (x axis) of the domain to be selected for the subset of data.

    Returns
    -------
    list[float]
        [x,y] array of the subset of data between xmin and xmax.

    """
    if xmin != None:
        index_xmin = (np.abs(x-xmin)).argmin()
        index_xmax = (np.abs(x-xmax)).argmin()
        if index_xmin>index_xmax:
            a=index_xmin
            index_xmin=index_xmax
            index_xmax=a
        return [x[index_xmin:index_xmax], y[index_xmin:index_xmax]]
    else:
        return [x,y]


@tool
def RMS(spec, xmin=None,xmax=None) -> float:
    """
    Get RMS from y data in [x,y] Python array.

    Parameters
    ----------
    spec : list[float]
        Python list [x,y] containing the spectrum. 

    xmin : float
        Minimum velocity value (x axis) of the domain to be used to calculate the RMS.

    xmax : float
        Maximum velocity value (x axis) of the domain to be used to calculate the RMS.

    Returns
    -------
    float
        RMS value.

    """
    vel_array = np.array(spec[0])
    Snu_array = np.array(spec[1])
    if xmin==None: xmin = vel_array[0]
    if xmax==None: xmax = vel_array[-1]
    spec_subdata = getsubdata(vel_array,Snu_array,xmin,xmax)

    x = spec_subdata[0]
    y = spec_subdata[1]
    rms = np.std(y)
    print('\t RMS = %1.3e (%s)' % (rms,ylabel))

    return rms



