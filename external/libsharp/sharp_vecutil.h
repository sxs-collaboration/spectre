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

/*! \file sharp_vecutil.h
 *  Functionality related to vector instruction support
 *
 *  Copyright (C) 2012,2013 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef SHARP_VECUTIL_H
#define SHARP_VECUTIL_H

#ifndef VLEN

#if (defined (__MIC__))
#define VLEN 8
#elif (defined (__AVX__))
#define VLEN 4
#elif (defined (__SSE2__))
#define VLEN 2
#else
#define VLEN 1
#endif

#endif

#if (VLEN==1)
#define VLEN_s 1
#else
#define VLEN_s (2*VLEN)
#endif

#ifndef USE_FMA4
#ifdef __FMA4__
#define USE_FMA4 1
#else
#define USE_FMA4 0
#endif
#endif

#endif
