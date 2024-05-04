/*
 *  This file is part of libsharp.
 *
 *  libsharp is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libsharp is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file sharp_mpi.h
 *  Interface for the spherical transform library with MPI support.
 *
 *  Copyright (C) 2011,2012 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifndef PLANCK_SHARP_MPI_H
#define PLANCK_SHARP_MPI_H

#include <mpi.h>
#include "sharp_lowlevel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! Performs an MPI parallel libsharp SHT job. The interface deliberately does
  not use the C99 "complex" data type, in order to be callable from C.
  \param comm the MPI communicator to be used for this SHT
  \param type the type of SHT
  \param spin the spin of the quantities to be transformed
  \param alm contains pointers to the a_lm coefficients. If \a spin==0,
    alm[0] points to the a_lm of the first SHT, alm[1] to those of the second
    etc. If \a spin>0, alm[0] and alm[1] point to the a_lm of the first SHT,
    alm[2] and alm[3] to those of the second, etc. The exact data type of \a alm
    depends on whether the SHARP_DP flag is set.
  \param map contains pointers to the maps. If \a spin==0,
    map[0] points to the map of the first SHT, map[1] to that of the second
    etc. If \a spin>0, or \a type is SHARP_ALM2MAP_DERIV1, map[0] and map[1]
    point to the maps of the first SHT, map[2] and map[3] to those of the
    second, etc. The exact data type of \a map depends on whether the SHARP_DP
    flag is set.
  \param geom_info A \c sharp_geom_info object compatible with the provided
    \a map arrays. The total map geometry is the union of all \a geom_info
    objects over the participating MPI tasks.
  \param alm_info A \c sharp_alm_info object compatible with the provided
    \a alm arrays. All \c m values from 0 to some \c mmax<=lmax must be present
    exactly once in the union of all \a alm_info objects over the participating
    MPI tasks.
  \param ntrans the number of simultaneous SHTs
  \param flags See sharp_jobflags. In particular, if SHARP_DP is set, then
    \a alm is expected to have the type "complex double **" and \a map is
    expected to have the type "double **"; otherwise, the expected
    types are "complex float **" and "float **", respectively.
  \param time If not NULL, the wall clock time required for this SHT
    (in seconds) will be written here.
  \param opcnt If not NULL, a conservative estimate of the total floating point
    operation count for this SHT will be written here. */
void sharp_execute_mpi (MPI_Comm comm, sharp_jobtype type, int spin,
  void *alm, void *map, const sharp_geom_info *geom_info,
  const sharp_alm_info *alm_info, int ntrans, int flags, double *time,
  unsigned long long *opcnt);

#ifdef __cplusplus
}
#endif

#endif
