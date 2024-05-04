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

/*! \file memusage.h
 *  Functionality for measuring memory consumption
 *
 *  Copyright (C) 2012 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef PLANCK_MEMUSAGE_H
#define PLANCK_MEMUSAGE_H

#ifdef __cplusplus
extern "C" {
#endif

/*! Returns the current resident set size in bytes.
    \note Currently only supported on Linux. Returns -1 if unsupported. */
double residentSetSize(void);

/*! Returns the high water mark of the resident set size in bytes.
    \note Currently only supported on Linux. Returns -1 if unsupported. */
double VmHWM(void);

#ifdef __cplusplus
}
#endif

#endif
