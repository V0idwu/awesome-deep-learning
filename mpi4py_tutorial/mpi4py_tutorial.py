#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@Time    :   2021/11/08 10:39:55
@Author  :   Tianyi Wu 
@Contact :   wutianyi@hotmail.com
@File    :   mpi4py_tutorial.py
@Version :   1.0
@Desc    :   None
"""

# here put the import lib
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    data = range(10)
    comm.send(data, dest=1, tag=11)
    print("process {} send {}...".format(rank, data))
else:
    data = comm.recv(source=0, tag=11)
    print("process {} recv {}...".format(rank, data))