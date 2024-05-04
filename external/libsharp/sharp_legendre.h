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

/*! \file sharp_legendre.h
 *  Interface for the Legendre transform parts of the spherical transform library.
 *
 *  Copyright (C) 2015 University of Oslo
 *  \author Dag Sverre Seljebotn
 */

#ifndef SHARP_LEGENDRE_H
#define SHARP_LEGENDRE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NO_LEGENDRE

void sharp_legendre_transform(double *bl, double *recfac, ptrdiff_t lmax, double *x,
                              double *out, ptrdiff_t nx);
void sharp_legendre_transform_s(float *bl, float *recfac, ptrdiff_t lmax, float *x,
                                float *out, ptrdiff_t nx);
void sharp_legendre_transform_recfac(double *r, ptrdiff_t lmax);
void sharp_legendre_transform_recfac_s(float *r, ptrdiff_t lmax);

#endif

#ifdef __cplusplus
}
#endif

#endif
