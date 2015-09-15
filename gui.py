# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 22:30:39 2015

@author: User
"""
import re
import numpy as np
import matplotlib.pyplot as mpl
import sys
if 'C:\Anaconda\Lib\site-packages' not in sys.path :
    sys.path.append('C:\Anaconda\Lib\site-packages')
import os
if 'D:\Gal\Work' not in sys.path:
    sys.path.append('D:\Gal\Work')
import classes4NBA as c
import technical_functions_NBA as tf
import research_functions_NBA as rf
from Tkinter import *
from ttk import *

def check():
    print 'check is done!'

def main_window(title=''):
    root = Tk()
    root.title(title)
    frame1 = Frame(root)
    CheckButton = Button(frame1,text='check',command=check)
    main_menu = Menu(frame1)
    root.config(menu=main_menu)

    players_menu = Menu(main_menu)
    main_menu.add_cascade(label='Players', menu=players_menu)
    players_menu.add_command(label='check', command=check)
    players_menu.add_command(label='exit', command=frame1.quit)

    CheckButton.pack(side=RIGHT)
    frame1.pack()
    return root
    
a = main_window('1')

cb = Combobox(a)
cb.
cb.pack()
b = main_window('2')
a.mainloop()

