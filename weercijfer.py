# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:06:14 2016

@author: gebruiker
"""

import pandas as pandas
import numpy as np


def read():
    knmi2 = pandas.read_csv('Bewerken2.csv', sep=',')
    knmi2['YYYYMMDD'] = pandas.to_datetime(knmi2['YYYYMMDD'], format='%Y%m%d')
    knmi2['YYYYMMDD'] = knmi2['YYYYMMDD'].dt.strftime('%Y-%m-%d')    
    knmi2 = np.matrix(knmi2)
    return knmi2


def rain(regen):
    regen = regen*6
    if regen < 10:
        return 0
    if regen < 90:
        return 1
    if regen < 300:
        return 2
    if regen < 500:
        return 3
    if regen >= 500:
        return 4


def dekkingsgraad(bewolking):
    if bewolking < 2:
        return 0
    if bewolking < 5:
        return 1
    if bewolking < 7:
        return 2
    if bewolking >= 7:
        return 3


def beaufort(wind):
    if wind < 34:
        return 0
    if wind < 55:
        return 1
    if wind < 108:
        return 2
    if wind >= 108:
        return 3


def weercijfer(wind, regen, bewolking):
    grade = 10
    grade = grade - rain(regen)
    grade = grade - dekkingsgraad(bewolking)
    grade = grade - beaufort(wind)

    return grade


def cijfer(matrix):
    cijferList = [['datum', 'cijfer']]
    for elem in matrix:
        elem = np.ravel(elem)
        datum = elem[1]
        wind = int(elem[4])
        regen = int(elem[21])
        bewolking = int(elem[-7])
        grade = weercijfer(wind, regen, bewolking)
        cijferList.append([datum, grade])
    return cijferList

if __name__ == "__main__":

    knmi = read()
    iets = np.ravel(knmi[-1])
    cijferList = cijfer(knmi)
    df = pandas.DataFrame(cijferList)
    df.to_csv('datadump.csv', sep=',')
