/* DO NOT EDIT. md5sum of source: a8c5c18a7a19c378187dbf461d12eb5c *//*

    NOTE NOTE NOTE

    This file is edited in sharp_legendre.c.in which is then preprocessed.
    Do not make manual  modifications to sharp_legendre.c.

    NOTE NOTE NOTE

*/


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

/*! \file sharp_legendre.c.in
 *
 *  Copyright (C) 2015 University of Oslo
 *  \author Dag Sverre Seljebotn
 */

#ifndef NO_LEGENDRE
#if (VLEN==8)
#error This code is not tested with MIC; please compile with -DNO_LEGENDRE
/* ...or test it (it probably works) and remove this check */
#endif

#ifndef SHARP_LEGENDRE_CS
#define SHARP_LEGENDRE_CS 4
#endif

#define MAX_CS 6
#if (SHARP_LEGENDRE_CS > MAX_CS)
#error (SHARP_LEGENDRE_CS > MAX_CS)
#endif

#include "sharp_legendre.h"
#include "sharp_vecsupport.h"

#include <stdlib.h>



static void legendre_transform_vec1(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(1) * VLEN],
                                              double out[(1) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
}

static void legendre_transform_vec2(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(2) * VLEN],
                                              double out[(2) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    x1 = vloadu(xarr + 1 * VLEN);
    Pm1_1 = vload(1.0);
    P_1 = x1;
    b = vload(*bl);
    y1 = vmul(Pm1_1, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    
    vfmaeq(y1, P_1, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul(x1, Pm1_1);
        W2 = W1;
        W2 = vsub(W2, Pm2_1);
        P_1 = W1;
        vfmaeq(P_1, W2, R);
        vfmaeq(y1, P_1, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
    vstoreu(out + 1 * VLEN, y1);
    
}

static void legendre_transform_vec3(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(3) * VLEN],
                                              double out[(3) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    x1 = vloadu(xarr + 1 * VLEN);
    Pm1_1 = vload(1.0);
    P_1 = x1;
    b = vload(*bl);
    y1 = vmul(Pm1_1, b);
    
    x2 = vloadu(xarr + 2 * VLEN);
    Pm1_2 = vload(1.0);
    P_2 = x2;
    b = vload(*bl);
    y2 = vmul(Pm1_2, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    
    vfmaeq(y1, P_1, b);
    
    vfmaeq(y2, P_2, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul(x1, Pm1_1);
        W2 = W1;
        W2 = vsub(W2, Pm2_1);
        P_1 = W1;
        vfmaeq(P_1, W2, R);
        vfmaeq(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul(x2, Pm1_2);
        W2 = W1;
        W2 = vsub(W2, Pm2_2);
        P_2 = W1;
        vfmaeq(P_2, W2, R);
        vfmaeq(y2, P_2, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
    vstoreu(out + 1 * VLEN, y1);
    
    vstoreu(out + 2 * VLEN, y2);
    
}

static void legendre_transform_vec4(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(4) * VLEN],
                                              double out[(4) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    x1 = vloadu(xarr + 1 * VLEN);
    Pm1_1 = vload(1.0);
    P_1 = x1;
    b = vload(*bl);
    y1 = vmul(Pm1_1, b);
    
    x2 = vloadu(xarr + 2 * VLEN);
    Pm1_2 = vload(1.0);
    P_2 = x2;
    b = vload(*bl);
    y2 = vmul(Pm1_2, b);
    
    x3 = vloadu(xarr + 3 * VLEN);
    Pm1_3 = vload(1.0);
    P_3 = x3;
    b = vload(*bl);
    y3 = vmul(Pm1_3, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    
    vfmaeq(y1, P_1, b);
    
    vfmaeq(y2, P_2, b);
    
    vfmaeq(y3, P_3, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul(x1, Pm1_1);
        W2 = W1;
        W2 = vsub(W2, Pm2_1);
        P_1 = W1;
        vfmaeq(P_1, W2, R);
        vfmaeq(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul(x2, Pm1_2);
        W2 = W1;
        W2 = vsub(W2, Pm2_2);
        P_2 = W1;
        vfmaeq(P_2, W2, R);
        vfmaeq(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul(x3, Pm1_3);
        W2 = W1;
        W2 = vsub(W2, Pm2_3);
        P_3 = W1;
        vfmaeq(P_3, W2, R);
        vfmaeq(y3, P_3, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
    vstoreu(out + 1 * VLEN, y1);
    
    vstoreu(out + 2 * VLEN, y2);
    
    vstoreu(out + 3 * VLEN, y3);
    
}

static void legendre_transform_vec5(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(5) * VLEN],
                                              double out[(5) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv P_4, Pm1_4, Pm2_4, x4, y4;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    x1 = vloadu(xarr + 1 * VLEN);
    Pm1_1 = vload(1.0);
    P_1 = x1;
    b = vload(*bl);
    y1 = vmul(Pm1_1, b);
    
    x2 = vloadu(xarr + 2 * VLEN);
    Pm1_2 = vload(1.0);
    P_2 = x2;
    b = vload(*bl);
    y2 = vmul(Pm1_2, b);
    
    x3 = vloadu(xarr + 3 * VLEN);
    Pm1_3 = vload(1.0);
    P_3 = x3;
    b = vload(*bl);
    y3 = vmul(Pm1_3, b);
    
    x4 = vloadu(xarr + 4 * VLEN);
    Pm1_4 = vload(1.0);
    P_4 = x4;
    b = vload(*bl);
    y4 = vmul(Pm1_4, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    
    vfmaeq(y1, P_1, b);
    
    vfmaeq(y2, P_2, b);
    
    vfmaeq(y3, P_3, b);
    
    vfmaeq(y4, P_4, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul(x1, Pm1_1);
        W2 = W1;
        W2 = vsub(W2, Pm2_1);
        P_1 = W1;
        vfmaeq(P_1, W2, R);
        vfmaeq(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul(x2, Pm1_2);
        W2 = W1;
        W2 = vsub(W2, Pm2_2);
        P_2 = W1;
        vfmaeq(P_2, W2, R);
        vfmaeq(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul(x3, Pm1_3);
        W2 = W1;
        W2 = vsub(W2, Pm2_3);
        P_3 = W1;
        vfmaeq(P_3, W2, R);
        vfmaeq(y3, P_3, b);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul(x4, Pm1_4);
        W2 = W1;
        W2 = vsub(W2, Pm2_4);
        P_4 = W1;
        vfmaeq(P_4, W2, R);
        vfmaeq(y4, P_4, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
    vstoreu(out + 1 * VLEN, y1);
    
    vstoreu(out + 2 * VLEN, y2);
    
    vstoreu(out + 3 * VLEN, y3);
    
    vstoreu(out + 4 * VLEN, y4);
    
}

static void legendre_transform_vec6(double *recfacs, double *bl, ptrdiff_t lmax,
                                              double xarr[(6) * VLEN],
                                              double out[(6) * VLEN]) {
    
    Tv P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv P_4, Pm1_4, Pm2_4, x4, y4;
    
    Tv P_5, Pm1_5, Pm2_5, x5, y5;
    
    Tv W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu(xarr + 0 * VLEN);
    Pm1_0 = vload(1.0);
    P_0 = x0;
    b = vload(*bl);
    y0 = vmul(Pm1_0, b);
    
    x1 = vloadu(xarr + 1 * VLEN);
    Pm1_1 = vload(1.0);
    P_1 = x1;
    b = vload(*bl);
    y1 = vmul(Pm1_1, b);
    
    x2 = vloadu(xarr + 2 * VLEN);
    Pm1_2 = vload(1.0);
    P_2 = x2;
    b = vload(*bl);
    y2 = vmul(Pm1_2, b);
    
    x3 = vloadu(xarr + 3 * VLEN);
    Pm1_3 = vload(1.0);
    P_3 = x3;
    b = vload(*bl);
    y3 = vmul(Pm1_3, b);
    
    x4 = vloadu(xarr + 4 * VLEN);
    Pm1_4 = vload(1.0);
    P_4 = x4;
    b = vload(*bl);
    y4 = vmul(Pm1_4, b);
    
    x5 = vloadu(xarr + 5 * VLEN);
    Pm1_5 = vload(1.0);
    P_5 = x5;
    b = vload(*bl);
    y5 = vmul(Pm1_5, b);
    
    
    b = vload(*(bl + 1));
    
    vfmaeq(y0, P_0, b);
    
    vfmaeq(y1, P_1, b);
    
    vfmaeq(y2, P_2, b);
    
    vfmaeq(y3, P_3, b);
    
    vfmaeq(y4, P_4, b);
    
    vfmaeq(y5, P_5, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload(*(bl + l));
        R = vload(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul(x0, Pm1_0);
        W2 = W1;
        W2 = vsub(W2, Pm2_0);
        P_0 = W1;
        vfmaeq(P_0, W2, R);
        vfmaeq(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul(x1, Pm1_1);
        W2 = W1;
        W2 = vsub(W2, Pm2_1);
        P_1 = W1;
        vfmaeq(P_1, W2, R);
        vfmaeq(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul(x2, Pm1_2);
        W2 = W1;
        W2 = vsub(W2, Pm2_2);
        P_2 = W1;
        vfmaeq(P_2, W2, R);
        vfmaeq(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul(x3, Pm1_3);
        W2 = W1;
        W2 = vsub(W2, Pm2_3);
        P_3 = W1;
        vfmaeq(P_3, W2, R);
        vfmaeq(y3, P_3, b);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul(x4, Pm1_4);
        W2 = W1;
        W2 = vsub(W2, Pm2_4);
        P_4 = W1;
        vfmaeq(P_4, W2, R);
        vfmaeq(y4, P_4, b);
        
        Pm2_5 = Pm1_5; Pm1_5 = P_5;
        W1 = vmul(x5, Pm1_5);
        W2 = W1;
        W2 = vsub(W2, Pm2_5);
        P_5 = W1;
        vfmaeq(P_5, W2, R);
        vfmaeq(y5, P_5, b);
        

    }
    
    vstoreu(out + 0 * VLEN, y0);
    
    vstoreu(out + 1 * VLEN, y1);
    
    vstoreu(out + 2 * VLEN, y2);
    
    vstoreu(out + 3 * VLEN, y3);
    
    vstoreu(out + 4 * VLEN, y4);
    
    vstoreu(out + 5 * VLEN, y5);
    
}



static void legendre_transform_vec1_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(1) * VLEN_s],
                                              float out[(1) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
}

static void legendre_transform_vec2_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(2) * VLEN_s],
                                              float out[(2) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vload_s(1.0);
    P_1 = x1;
    b = vload_s(*bl);
    y1 = vmul_s(Pm1_1, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    
    vfmaeq_s(y1, P_1, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = W1;
        vfmaeq_s(P_1, W2, R);
        vfmaeq_s(y1, P_1, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
}

static void legendre_transform_vec3_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(3) * VLEN_s],
                                              float out[(3) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv_s P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vload_s(1.0);
    P_1 = x1;
    b = vload_s(*bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vload_s(1.0);
    P_2 = x2;
    b = vload_s(*bl);
    y2 = vmul_s(Pm1_2, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    
    vfmaeq_s(y1, P_1, b);
    
    vfmaeq_s(y2, P_2, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = W1;
        vfmaeq_s(P_1, W2, R);
        vfmaeq_s(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = W1;
        vfmaeq_s(P_2, W2, R);
        vfmaeq_s(y2, P_2, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
}

static void legendre_transform_vec4_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(4) * VLEN_s],
                                              float out[(4) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv_s P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv_s P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vload_s(1.0);
    P_1 = x1;
    b = vload_s(*bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vload_s(1.0);
    P_2 = x2;
    b = vload_s(*bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vload_s(1.0);
    P_3 = x3;
    b = vload_s(*bl);
    y3 = vmul_s(Pm1_3, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    
    vfmaeq_s(y1, P_1, b);
    
    vfmaeq_s(y2, P_2, b);
    
    vfmaeq_s(y3, P_3, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = W1;
        vfmaeq_s(P_1, W2, R);
        vfmaeq_s(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = W1;
        vfmaeq_s(P_2, W2, R);
        vfmaeq_s(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = W1;
        vfmaeq_s(P_3, W2, R);
        vfmaeq_s(y3, P_3, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
}

static void legendre_transform_vec5_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(5) * VLEN_s],
                                              float out[(5) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv_s P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv_s P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv_s P_4, Pm1_4, Pm2_4, x4, y4;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vload_s(1.0);
    P_1 = x1;
    b = vload_s(*bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vload_s(1.0);
    P_2 = x2;
    b = vload_s(*bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vload_s(1.0);
    P_3 = x3;
    b = vload_s(*bl);
    y3 = vmul_s(Pm1_3, b);
    
    x4 = vloadu_s(xarr + 4 * VLEN_s);
    Pm1_4 = vload_s(1.0);
    P_4 = x4;
    b = vload_s(*bl);
    y4 = vmul_s(Pm1_4, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    
    vfmaeq_s(y1, P_1, b);
    
    vfmaeq_s(y2, P_2, b);
    
    vfmaeq_s(y3, P_3, b);
    
    vfmaeq_s(y4, P_4, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = W1;
        vfmaeq_s(P_1, W2, R);
        vfmaeq_s(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = W1;
        vfmaeq_s(P_2, W2, R);
        vfmaeq_s(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = W1;
        vfmaeq_s(P_3, W2, R);
        vfmaeq_s(y3, P_3, b);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_s(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_4);
        P_4 = W1;
        vfmaeq_s(P_4, W2, R);
        vfmaeq_s(y4, P_4, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
    vstoreu_s(out + 4 * VLEN_s, y4);
    
}

static void legendre_transform_vec6_s(float *recfacs, float *bl, ptrdiff_t lmax,
                                              float xarr[(6) * VLEN_s],
                                              float out[(6) * VLEN_s]) {
    
    Tv_s P_0, Pm1_0, Pm2_0, x0, y0;
    
    Tv_s P_1, Pm1_1, Pm2_1, x1, y1;
    
    Tv_s P_2, Pm1_2, Pm2_2, x2, y2;
    
    Tv_s P_3, Pm1_3, Pm2_3, x3, y3;
    
    Tv_s P_4, Pm1_4, Pm2_4, x4, y4;
    
    Tv_s P_5, Pm1_5, Pm2_5, x5, y5;
    
    Tv_s W1, W2, b, R;
    ptrdiff_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vload_s(1.0);
    P_0 = x0;
    b = vload_s(*bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vload_s(1.0);
    P_1 = x1;
    b = vload_s(*bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vload_s(1.0);
    P_2 = x2;
    b = vload_s(*bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vload_s(1.0);
    P_3 = x3;
    b = vload_s(*bl);
    y3 = vmul_s(Pm1_3, b);
    
    x4 = vloadu_s(xarr + 4 * VLEN_s);
    Pm1_4 = vload_s(1.0);
    P_4 = x4;
    b = vload_s(*bl);
    y4 = vmul_s(Pm1_4, b);
    
    x5 = vloadu_s(xarr + 5 * VLEN_s);
    Pm1_5 = vload_s(1.0);
    P_5 = x5;
    b = vload_s(*bl);
    y5 = vmul_s(Pm1_5, b);
    
    
    b = vload_s(*(bl + 1));
    
    vfmaeq_s(y0, P_0, b);
    
    vfmaeq_s(y1, P_1, b);
    
    vfmaeq_s(y2, P_2, b);
    
    vfmaeq_s(y3, P_3, b);
    
    vfmaeq_s(y4, P_4, b);
    
    vfmaeq_s(y5, P_5, b);
    

    for (l = 2; l <= lmax; ++l) {
        b = vload_s(*(bl + l));
        R = vload_s(*(recfacs + l));
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = W1;
        vfmaeq_s(P_0, W2, R);
        vfmaeq_s(y0, P_0, b);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = W1;
        vfmaeq_s(P_1, W2, R);
        vfmaeq_s(y1, P_1, b);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = W1;
        vfmaeq_s(P_2, W2, R);
        vfmaeq_s(y2, P_2, b);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = W1;
        vfmaeq_s(P_3, W2, R);
        vfmaeq_s(y3, P_3, b);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_s(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_4);
        P_4 = W1;
        vfmaeq_s(P_4, W2, R);
        vfmaeq_s(y4, P_4, b);
        
        Pm2_5 = Pm1_5; Pm1_5 = P_5;
        W1 = vmul_s(x5, Pm1_5);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_5);
        P_5 = W1;
        vfmaeq_s(P_5, W2, R);
        vfmaeq_s(y5, P_5, b);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
    vstoreu_s(out + 4 * VLEN_s, y4);
    
    vstoreu_s(out + 5 * VLEN_s, y5);
    
}





void sharp_legendre_transform_recfac(double *r, ptrdiff_t lmax) {
    /* (l - 1) / l, for l >= 2 */
    ptrdiff_t l;
    r[0] = 0;
    r[1] = 1;
    for (l = 2; l <= lmax; ++l) {
        r[l] = (double)(l - 1) / (double)l;
    }
}

void sharp_legendre_transform_recfac_s(float *r, ptrdiff_t lmax) {
    /* (l - 1) / l, for l >= 2 */
    ptrdiff_t l;
    r[0] = 0;
    r[1] = 1;
    for (l = 2; l <= lmax; ++l) {
        r[l] = (float)(l - 1) / (float)l;
    }
}


/*
  Compute sum_l b_l P_l(x_i) for all i. 
 */

#define LEN (SHARP_LEGENDRE_CS * VLEN)
#define LEN_s (SHARP_LEGENDRE_CS * VLEN_s)


void sharp_legendre_transform(double *bl,
                                   double *recfac,
                                   ptrdiff_t lmax,
                                   double *x, double *out, ptrdiff_t nx) {
    double xchunk[MAX_CS * VLEN], outchunk[MAX_CS * LEN];
    int compute_recfac;
    ptrdiff_t i, j, len;

    compute_recfac = (recfac == NULL);
    if (compute_recfac) {
        recfac = malloc(sizeof(double) * (lmax + 1));
        sharp_legendre_transform_recfac(recfac, lmax);
    }

    for (j = 0; j != LEN; ++j) xchunk[j] = 0;

    for (i = 0; i < nx; i += LEN) {
        len = (i + (LEN) <= nx) ? (LEN) : (nx - i);
        for (j = 0; j != len; ++j) xchunk[j] = x[i + j];
        switch ((len + VLEN - 1) / VLEN) {
          case 6: legendre_transform_vec6(recfac, bl, lmax, xchunk, outchunk); break;
          case 5: legendre_transform_vec5(recfac, bl, lmax, xchunk, outchunk); break;
          case 4: legendre_transform_vec4(recfac, bl, lmax, xchunk, outchunk); break;
          case 3: legendre_transform_vec3(recfac, bl, lmax, xchunk, outchunk); break;
          case 2: legendre_transform_vec2(recfac, bl, lmax, xchunk, outchunk); break;
          case 1:
          case 0:
              legendre_transform_vec1(recfac, bl, lmax, xchunk, outchunk); break;
        }
        for (j = 0; j != len; ++j) out[i + j] = outchunk[j];
    }
    if (compute_recfac) {
        free(recfac);
    }
}

void sharp_legendre_transform_s(float *bl,
                                   float *recfac,
                                   ptrdiff_t lmax,
                                   float *x, float *out, ptrdiff_t nx) {
    float xchunk[MAX_CS * VLEN_s], outchunk[MAX_CS * LEN_s];
    int compute_recfac;
    ptrdiff_t i, j, len;

    compute_recfac = (recfac == NULL);
    if (compute_recfac) {
        recfac = malloc(sizeof(float) * (lmax + 1));
        sharp_legendre_transform_recfac_s(recfac, lmax);
    }

    for (j = 0; j != LEN_s; ++j) xchunk[j] = 0;

    for (i = 0; i < nx; i += LEN_s) {
        len = (i + (LEN_s) <= nx) ? (LEN_s) : (nx - i);
        for (j = 0; j != len; ++j) xchunk[j] = x[i + j];
        switch ((len + VLEN_s - 1) / VLEN_s) {
          case 6: legendre_transform_vec6_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 5: legendre_transform_vec5_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 4: legendre_transform_vec4_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 3: legendre_transform_vec3_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 2: legendre_transform_vec2_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 1:
          case 0:
              legendre_transform_vec1_s(recfac, bl, lmax, xchunk, outchunk); break;
        }
        for (j = 0; j != len; ++j) out[i + j] = outchunk[j];
    }
    if (compute_recfac) {
        free(recfac);
    }
}


#endif