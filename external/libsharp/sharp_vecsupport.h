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

/*  \file sharp_vecsupport.h
 *  Convenience functions for vector arithmetics
 *
 *  Copyright (C) 2012,2013 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#ifndef SHARP_VECSUPPORT_H
#define SHARP_VECSUPPORT_H

#include <math.h>
#include "sharp_vecutil.h"

typedef double Ts;

#if (VLEN==1)

typedef double Tv;
typedef float Tv_s;
typedef int Tm;

#define vadd(a,b) ((a)+(b))
#define vadd_s(a,b) ((a)+(b))
#define vaddeq(a,b) ((a)+=(b))
#define vaddeq_mask(mask,a,b) if (mask) (a)+=(b);
#define vsub(a,b) ((a)-(b))
#define vsub_s(a,b) ((a)-(b))
#define vsubeq(a,b) ((a)-=(b))
#define vsubeq_mask(mask,a,b) if (mask) (a)-=(b);
#define vmul(a,b) ((a)*(b))
#define vmul_s(a,b) ((a)*(b))
#define vmuleq(a,b) ((a)*=(b))
#define vmuleq_mask(mask,a,b) if (mask) (a)*=(b);
#define vfmaeq(a,b,c) ((a)+=(b)*(c))
#define vfmaeq_s(a,b,c) ((a)+=(b)*(c))
#define vfmseq(a,b,c) ((a)-=(b)*(c))
#define vfmaaeq(a,b,c,d,e) ((a)+=(b)*(c)+(d)*(e))
#define vfmaseq(a,b,c,d,e) ((a)+=(b)*(c)-(d)*(e))
#define vneg(a) (-(a))
#define vload(a) (a)
#define vload_s(a) (a)
#define vloadu(p) (*(p))
#define vloadu_s(p) (*(p))
#define vabs(a) fabs(a)
#define vsqrt(a) sqrt(a)
#define vlt(a,b) ((a)<(b))
#define vgt(a,b) ((a)>(b))
#define vge(a,b) ((a)>=(b))
#define vne(a,b) ((a)!=(b))
#define vand_mask(a,b) ((a)&&(b))
#define vstoreu(p, a) (*(p)=a)
#define vstoreu_s(p, a) (*(p)=a)

static inline Tv vmin (Tv a, Tv b) { return (a<b) ? a : b; }
static inline Tv vmax (Tv a, Tv b) { return (a>b) ? a : b; }

#define vanyTrue(a) (a)
#define vallTrue(a) (a)
#define vzero 0.
#define vone 1.

#endif

#if (VLEN==2)

#include <emmintrin.h>

#if defined (__SSE3__)
#include <pmmintrin.h>
#endif
#if defined (__SSE4_1__)
#include <smmintrin.h>
#endif

typedef __m128d Tv;
typedef __m128 Tv_s;
typedef __m128d Tm;

#if defined(__SSE4_1__)
#define vblend__(m,a,b) _mm_blendv_pd(b,a,m)
#else
static inline Tv vblend__(Tv m, Tv a, Tv b)
  { return _mm_or_pd(_mm_and_pd(a,m),_mm_andnot_pd(m,b)); }
#endif
#define vzero _mm_setzero_pd()
#define vone _mm_set1_pd(1.)

#define vadd(a,b) _mm_add_pd(a,b)
#define vadd_s(a,b) _mm_add_ps(a,b)
#define vaddeq(a,b) a=_mm_add_pd(a,b)
#define vaddeq_mask(mask,a,b) a=_mm_add_pd(a,vblend__(mask,b,vzero))
#define vsub(a,b) _mm_sub_pd(a,b)
#define vsub_s(a,b) _mm_sub_ps(a,b)
#define vsubeq(a,b) a=_mm_sub_pd(a,b)
#define vsubeq_mask(mask,a,b) a=_mm_sub_pd(a,vblend__(mask,b,vzero))
#define vmul(a,b) _mm_mul_pd(a,b)
#define vmul_s(a,b) _mm_mul_ps(a,b)
#define vmuleq(a,b) a=_mm_mul_pd(a,b)
#define vmuleq_mask(mask,a,b) a=_mm_mul_pd(a,vblend__(mask,b,vone))
#define vfmaeq(a,b,c) a=_mm_add_pd(a,_mm_mul_pd(b,c))
#define vfmaeq_s(a,b,c) a=_mm_add_ps(a,_mm_mul_ps(b,c))
#define vfmseq(a,b,c) a=_mm_sub_pd(a,_mm_mul_pd(b,c))
#define vfmaaeq(a,b,c,d,e) \
  a=_mm_add_pd(a,_mm_add_pd(_mm_mul_pd(b,c),_mm_mul_pd(d,e)))
#define vfmaseq(a,b,c,d,e) \
  a=_mm_add_pd(a,_mm_sub_pd(_mm_mul_pd(b,c),_mm_mul_pd(d,e)))
#define vneg(a) _mm_xor_pd(_mm_set1_pd(-0.),a)
#define vload(a) _mm_set1_pd(a)
#define vload_s(a) _mm_set1_ps(a)
#define vabs(a) _mm_andnot_pd(_mm_set1_pd(-0.),a)
#define vsqrt(a) _mm_sqrt_pd(a)
#define vlt(a,b) _mm_cmplt_pd(a,b)
#define vgt(a,b) _mm_cmpgt_pd(a,b)
#define vge(a,b) _mm_cmpge_pd(a,b)
#define vne(a,b) _mm_cmpneq_pd(a,b)
#define vand_mask(a,b) _mm_and_pd(a,b)
#define vmin(a,b) _mm_min_pd(a,b)
#define vmax(a,b) _mm_max_pd(a,b);
#define vanyTrue(a) (_mm_movemask_pd(a)!=0)
#define vallTrue(a) (_mm_movemask_pd(a)==3)
#define vloadu(p) _mm_loadu_pd(p)
#define vloadu_s(p) _mm_loadu_ps(p)
#define vstoreu(p, v) _mm_storeu_pd(p, v)
#define vstoreu_s(p, v) _mm_storeu_ps(p, v)

#endif

#if (VLEN==4)

#include <immintrin.h>
#if (USE_FMA4)
#include <x86intrin.h>
#endif

typedef __m256d Tv;
typedef __m256 Tv_s;
typedef __m256d Tm;

#define vblend__(m,a,b) _mm256_blendv_pd(b,a,m)
#define vzero _mm256_setzero_pd()
#define vone _mm256_set1_pd(1.)

#define vadd(a,b) _mm256_add_pd(a,b)
#define vadd_s(a,b) _mm256_add_ps(a,b)
#define vaddeq(a,b) a=_mm256_add_pd(a,b)
#define vaddeq_mask(mask,a,b) a=_mm256_add_pd(a,vblend__(mask,b,vzero))
#define vsub(a,b) _mm256_sub_pd(a,b)
#define vsub_s(a,b) _mm256_sub_ps(a,b)
#define vsubeq(a,b) a=_mm256_sub_pd(a,b)
#define vsubeq_mask(mask,a,b) a=_mm256_sub_pd(a,vblend__(mask,b,vzero))
#define vmul(a,b) _mm256_mul_pd(a,b)
#define vmul_s(a,b) _mm256_mul_ps(a,b)
#define vmuleq(a,b) a=_mm256_mul_pd(a,b)
#define vmuleq_mask(mask,a,b) a=_mm256_mul_pd(a,vblend__(mask,b,vone))
#if (USE_FMA4)
#define vfmaeq(a,b,c) a=_mm256_macc_pd(b,c,a)
#define vfmaeq_s(a,b,c) a=_mm256_macc_ps(b,c,a)
#define vfmseq(a,b,c) a=_mm256_nmacc_pd(b,c,a)
#define vfmaaeq(a,b,c,d,e) a=_mm256_macc_pd(d,e,_mm256_macc_pd(b,c,a))
#define vfmaseq(a,b,c,d,e) a=_mm256_nmacc_pd(d,e,_mm256_macc_pd(b,c,a))
#else
#define vfmaeq(a,b,c) a=_mm256_add_pd(a,_mm256_mul_pd(b,c))
#define vfmaeq_s(a,b,c) a=_mm256_add_ps(a,_mm256_mul_ps(b,c))
#define vfmseq(a,b,c) a=_mm256_sub_pd(a,_mm256_mul_pd(b,c))
#define vfmaaeq(a,b,c,d,e) \
  a=_mm256_add_pd(a,_mm256_add_pd(_mm256_mul_pd(b,c),_mm256_mul_pd(d,e)))
#define vfmaseq(a,b,c,d,e) \
  a=_mm256_add_pd(a,_mm256_sub_pd(_mm256_mul_pd(b,c),_mm256_mul_pd(d,e)))
#endif
#define vneg(a) _mm256_xor_pd(_mm256_set1_pd(-0.),a)
#define vload(a) _mm256_set1_pd(a)
#define vload_s(a) _mm256_set1_ps(a)
#define vabs(a) _mm256_andnot_pd(_mm256_set1_pd(-0.),a)
#define vsqrt(a) _mm256_sqrt_pd(a)
#define vlt(a,b) _mm256_cmp_pd(a,b,_CMP_LT_OQ)
#define vgt(a,b) _mm256_cmp_pd(a,b,_CMP_GT_OQ)
#define vge(a,b) _mm256_cmp_pd(a,b,_CMP_GE_OQ)
#define vne(a,b) _mm256_cmp_pd(a,b,_CMP_NEQ_OQ)
#define vand_mask(a,b) _mm256_and_pd(a,b)
#define vmin(a,b) _mm256_min_pd(a,b)
#define vmax(a,b) _mm256_max_pd(a,b)
#define vanyTrue(a) (_mm256_movemask_pd(a)!=0)
#define vallTrue(a) (_mm256_movemask_pd(a)==15)

#define vloadu(p) _mm256_loadu_pd(p)
#define vloadu_s(p) _mm256_loadu_ps(p)
#define vstoreu(p, v) _mm256_storeu_pd(p, v)
#define vstoreu_s(p, v) _mm256_storeu_ps(p, v)

#endif

#if (VLEN==8)

#include <immintrin.h>

typedef __m512d Tv;
typedef __mmask8 Tm;

#define vadd(a,b) _mm512_add_pd(a,b)
#define vaddeq(a,b) a=_mm512_add_pd(a,b)
#define vaddeq_mask(mask,a,b) a=_mm512_mask_add_pd(a,mask,a,b);
#define vsub(a,b) _mm512_sub_pd(a,b)
#define vsubeq(a,b) a=_mm512_sub_pd(a,b)
#define vsubeq_mask(mask,a,b) a=_mm512_mask_sub_pd(a,mask,a,b);
#define vmul(a,b) _mm512_mul_pd(a,b)
#define vmuleq(a,b) a=_mm512_mul_pd(a,b)
#define vmuleq_mask(mask,a,b) a=_mm512_mask_mul_pd(a,mask,a,b);
#define vfmaeq(a,b,c) a=_mm512_fmadd_pd(b,c,a)
#define vfmseq(a,b,c) a=_mm512_fnmadd_pd(b,c,a)
#define vfmaaeq(a,b,c,d,e) a=_mm512_fmadd_pd(d,e,_mm512_fmadd_pd(b,c,a))
#define vfmaseq(a,b,c,d,e) a=_mm512_fnmadd_pd(d,e,_mm512_fmadd_pd(b,c,a))
#define vneg(a) _mm512_mul_pd(a,_mm512_set1_pd(-1.))
#define vload(a) _mm512_set1_pd(a)
#define vabs(a) (__m512d)_mm512_andnot_epi64((__m512i)_mm512_set1_pd(-0.),(__m512i)a)
#define vsqrt(a) _mm512_sqrt_pd(a)
#define vlt(a,b) _mm512_cmplt_pd_mask(a,b)
#define vgt(a,b) _mm512_cmpnle_pd_mask(a,b)
#define vge(a,b) _mm512_cmpnlt_pd_mask(a,b)
#define vne(a,b) _mm512_cmpneq_pd_mask(a,b)
#define vand_mask(a,b) ((a)&(b))
#define vmin(a,b) _mm512_min_pd(a,b)
#define vmax(a,b) _mm512_max_pd(a,b)
#define vanyTrue(a) (a!=0)
#define vallTrue(a) (a==255)

#define vzero _mm512_setzero_pd()
#define vone _mm512_set1_pd(1.)

#endif

#endif
