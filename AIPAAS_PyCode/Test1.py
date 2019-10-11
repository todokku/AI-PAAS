# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:51 2019

@author: tejas
"""
from Input import cMAPPS
import CMAPPS_ORM as orm

orm.create_session('postgresql://postgres:Candy@065@localhost:5432/postgres')