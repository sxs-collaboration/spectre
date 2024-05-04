/*
 *  This file is part of libsharp.
 *
 * Redistribution and use in source and binary forms, with or without
 * met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file sharp_legendre_table.h
 *  Interface for computing tables of the normalized associated Legendre transform
 *
 *  Copyright (C) 2017 Dag Sverre Seljebotn
 *  \author Dag Sverre Seljebotn
 *
 *  Note: This code was mainly copied from libpsht; only a small high-level wrapper added
 */

#ifndef SHARP_LEGENDRE_TABLE_H
#define SHARP_LEGENDRE_TABLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NO_LEGENDRE_TABLE


/*! Returns a table of the normalized associated Legendre polynomials. m is a single
    fixed argument and a table for multiple l and cos(theta) is provided.
    (Internally, sin(theta) is also used for part of the computation, making theta
    the most convenient argument.)

    NOTE: Support for spin-weighted Legendre functions is on the TODO-list. Only spin=0
    is supported now.

    \param m The m-value to compute a table for; must be >= 0
    \param spin The spin parameter; pass 0 for the regular associated Legendre functions.
                NOTE: This is present for future compatability, currently only 0 is supported.
    \param lmax A table will be provided for l = m .. lmax
    \param ntheta How many theta values to evaluate for
    \param theta Contiguous 1D array of theta values
    \param theta_stride See below
    \param l_stride See below
    \param spin_stride See below. "ispin" will always be 0 if spin==0, or 0 for positive spin
                       and 1 for the corresponding negative spin otherwise.
    \param out Contiguous 3D array that will receive the output. Each output entry
               is assigned to out[itheta * theta_stride + (l - m) * l_stride + ispin * spin_stride].
 */
void sharp_normalized_associated_legendre_table(
  ptrdiff_t m,
  int spin,
  ptrdiff_t lmax,
  ptrdiff_t ntheta,
  /* contiguous 1D array of theta values to compute for,
     contains ntheta values */
  double *theta,
  /* contiguous 2D array, in "theta-major ordering". Has `ntheta`
     rows and `ncols` columns. Indexed as out[itheta * ncols + (l - m)].
     If `ncols > lmax - m` then those entries are not accessed.
  */
  ptrdiff_t theta_stride,
  ptrdiff_t l_stride,
  ptrdiff_t spin_stride,
  double *out
);

#endif

#ifdef __cplusplus
}
#endif

#endif
