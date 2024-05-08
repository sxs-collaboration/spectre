/*
 *  This file is part of libc_utils.
 *
 *  libc_utils is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libc_utils is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libc_utils; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libc_utils is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file sharp_announce.c
 *  Banner for module startup
 *
 *  Copyright (C) 2012 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "sharp_announce.h"
#include "sharp_vecutil.h"

static void OpenMP_status(void)
  {
#ifndef _OPENMP
  printf("OpenMP: not supported by this binary\n");
#else
  int threads = omp_get_max_threads();
  if (threads>1)
    printf("OpenMP active: max. %d threads.\n",threads);
  else
    printf("OpenMP active, but running with 1 thread only.\n");
#endif
  }

static void MPI_status(void)
  {
#ifndef USE_MPI
  printf("MPI: not supported by this binary\n");
#else
  int tasks;
  MPI_Comm_size(MPI_COMM_WORLD,&tasks);
  if (tasks>1)
    printf("MPI active with %d tasks.\n",tasks);
  else
    printf("MPI active, but running with 1 task only.\n");
#endif
  }

static void vecmath_status(void)
  { printf("Supported vector length: %d\n",VLEN); }

void sharp_announce (const char *name)
  {
  size_t m, nlen=strlen(name);
  printf("\n+-");
  for (m=0; m<nlen; ++m) printf("-");
  printf("-+\n");
  printf("| %s |\n", name);
  printf("+-");
  for (m=0; m<nlen; ++m) printf("-");
  printf("-+\n\n");
  vecmath_status();
  OpenMP_status();
  MPI_status();
  printf("\n");
  }

void sharp_module_startup (const char *name, int argc, int argc_expected,
  const char *argv_expected, int verbose)
  {
  if (verbose) sharp_announce (name);
  if (argc==argc_expected) return;
  if (verbose) fprintf(stderr, "Usage: %s %s\n", name, argv_expected);
  exit(1);
  }
