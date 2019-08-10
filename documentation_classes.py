#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 22:38:31 2019

@author: konstantin
"""

# Python class documentation

def scope_test():
    def do_local():
        spam = "local spam"
        
    def do_nonlocal():
        nonlocal spam 
        spam = "non local spam"
        
    def do_global():
        global spam
        spam = "global spam"
              
        spam = "test spam"
        do_local()
        print("After local assignment:", spam)
        do_nonlocal()
        print("After nonlocal assignment:", spam)
        do_global()
        print("After global assignment:", spam)
        
    return


# Classes

# classes have to be executed, before they take effect, like functions
# classes contain statements, in practive mostly functions
# class defintions create new local namespaces
# class objects support two kinds of operations 
# attribute references and instantiations

class MyClass:
    """ A simple example class"""
    i = 12345
    
    def f(self):
        return "Hello World!"
    
MyClass.i = "Stay present"

MyClass.f = "sd"
MyClass.f() # attribute reference
MyClass.__doc__

x = MyClass() # instantiation
# creates empty object. Many classes like to create objects with instancex
# customized to a specific initial state








