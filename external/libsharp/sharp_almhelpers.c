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

/*! \file sharp_almhelpers.c
 *  Spherical transform library
 *
 *  Copyright (C) 2008-2013 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include "sharp_almhelpers.h"
#include "c_utils.h"

void sharp_make_triangular_alm_info (int lmax, int mmax, int stride,
  sharp_alm_info **alm_info)
  {
  sharp_alm_info *info = RALLOC(sharp_alm_info,1);
  info->lmax = lmax;
  info->nm = mmax+1;
  info->mval = RALLOC(int,mmax+1);
  info->mvstart = RALLOC(ptrdiff_t,mmax+1);
  info->stride = stride;
  info->flags = 0;
  ptrdiff_t tval = 2*lmax+1;
  for (ptrdiff_t m=0; m<=mmax; ++m)
    {
    info->mval[m] = m;
    info->mvstart[m] = stride*((m*(tval-m))>>1);
    }
  *alm_info = info;
  }

void sharp_make_rectangular_alm_info (int lmax, int mmax, int stride,
  sharp_alm_info **alm_info)
  {
  sharp_alm_info *info = RALLOC(sharp_alm_info,1);
  info->lmax = lmax;
  info->nm = mmax+1;
  info->mval = RALLOC(int,mmax+1);
  info->mvstart = RALLOC(ptrdiff_t,mmax+1);
  info->stride = stride;
  info->flags = 0;
  for (ptrdiff_t m=0; m<=mmax; ++m)
    {
    info->mval[m] = m;
    info->mvstart[m] = stride*m*(lmax+1);
    }
  *alm_info = info;
  }

void sharp_make_mmajor_real_packed_alm_info (int lmax, int stride,
  int nm, const int *ms, sharp_alm_info **alm_info)
  {
  ptrdiff_t idx;
  int f;
  sharp_alm_info *info = RALLOC(sharp_alm_info,1);
  info->lmax = lmax;
  info->nm = nm;
  info->mval = RALLOC(int,nm);
  info->mvstart = RALLOC(ptrdiff_t,nm);
  info->stride = stride;
  info->flags = SHARP_PACKED | SHARP_REAL_HARMONICS;
  idx = 0;  /* tracks the number of 'consumed' elements so far; need to correct by m */
  for (int im=0; im!=nm; ++im)
    {
    int m=(ms==NULL)?im:ms[im];
    f = (m==0) ? 1 : 2;
    info->mval[im] = m;
    info->mvstart[im] = stride * (idx - f * m);
    idx += f * (lmax + 1 - m);
    }
  *alm_info = info;
  }
