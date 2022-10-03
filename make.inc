# -*- Makefile -*-
# Copyright 2010-2011, Georgia Institute of Technology, USA.
# See COPYING for license.

CC = gcc
CFLAGS = -O3 -fopenmp #-march=armv8.2-a+sve -mtune=a64fx -DZFILL_CACHE_LINES

#CC = fcc -Kfast,openmp,largepage
# CFLAGS = -Kprefetch_sequential=soft -Kprefetch_line=8 -Kprefetch_line_L2=16 -Kzfill
# CFLAGS = -Kprefetch_sequential=soft -Kprefetch_line=9 -Kprefetch_line_L2=40 -Kzfill=18
# CFLAGS = -Kprefetch_sequential=soft -Kzfill
# CFLAGS = -Kzfill
LDLIBS = -lm -lrt

## Uncomment if you have libnuma and want a numactl-aware malloc.
# LDLIBS += -lnuma
# CPPFLAGS += -DHAVE_LIBNUMA

## Uncomment to use a file-backed mapping for large data.
# CPPFLAGS += -DUSE_MMAP_LARGE -DUSE_MMAP_LARGE_EXT

## Uncomment to build OpenMP examples.
# BUILD_OPENMP = Yes
# CFLAGS_OPENMP = -fopenmp

## Uncomment to build the MPI code using default compilers, etc.
# BUILD_MPI = Yes
