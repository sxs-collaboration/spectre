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

/*  \file sharp_testsuite.c
 * 
 *  Copyright (C) 2012-2013 Max-Planck-Society
 *  \author Martin Reinecke
 */

#include <stdio.h>
#include <string.h>
#ifdef USE_MPI
#include "mpi.h"
#include "sharp_mpi.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "sharp.h"
#include "sharp_internal.h"
#include "sharp_geomhelpers.h"
#include "sharp_almhelpers.h"
#include "c_utils.h"
#include "sharp_announce.h"
#include "memusage.h"
#include "sharp_vecsupport.h"

typedef complex double dcmplx;

int ntasks, mytask;

static double drand (double min, double max, int *state)
  {
  *state = (((*state) * 1103515245) + 12345) & 0x7fffffff;
  return min + (max-min)*(*state)/(0x7fffffff+1.0);
  }

static void random_alm (dcmplx *alm, sharp_alm_info *helper, int spin, int cnt)
  {
#pragma omp parallel
{
  int mi;
#pragma omp for schedule (dynamic,100)
  for (mi=0;mi<helper->nm; ++mi)
    {
    int m=helper->mval[mi];
    int state=1234567*cnt+8912*m; // random seed
    for (int l=m;l<=helper->lmax; ++l)
      {
      if ((l<spin)&&(m<spin))
        alm[sharp_alm_index(helper,l,mi)] = 0.;
      else
        {
        double rv = drand(-1,1,&state);
        double iv = (m==0) ? 0 : drand(-1,1,&state);
        alm[sharp_alm_index(helper,l,mi)] = rv+_Complex_I*iv;
        }
      }
    }
} // end of parallel region
  }

static unsigned long long totalops (unsigned long long val)
  {
#ifdef USE_MPI
  unsigned long long tmp;
  MPI_Allreduce (&val, &tmp,1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  return tmp;
#else
  return val;
#endif
  }

static double maxTime (double val)
  {
#ifdef USE_MPI
  double tmp;
  MPI_Allreduce (&val, &tmp,1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return tmp;
#else
  return val;
#endif
  }

static double allreduceSumDouble (double val)
  {
#ifdef USE_MPI
  double tmp;
  MPI_Allreduce (&val, &tmp,1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return tmp;
#else
  return val;
#endif
  }

static double totalMem()
  {
#ifdef USE_MPI
  double tmp, val=VmHWM();
  MPI_Allreduce (&val, &tmp,1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return tmp;
#else
  return VmHWM();
#endif
  }

#ifdef USE_MPI
static void reduce_alm_info(sharp_alm_info *ainfo)
  {
  int nmnew=0;
  ptrdiff_t ofs = 0;
  for (int i=mytask; i<ainfo->nm; i+=ntasks,++nmnew)
    {
    ainfo->mval[nmnew]=ainfo->mval[i];
    ainfo->mvstart[nmnew]=ofs-ainfo->mval[nmnew];
    ofs+=ainfo->lmax-ainfo->mval[nmnew]+1;
    }
  ainfo->nm=nmnew;
  }

static void reduce_geom_info(sharp_geom_info *ginfo)
  {
  int npairsnew=0;
  ptrdiff_t ofs = 0;
  for (int i=mytask; i<ginfo->npairs; i+=ntasks,++npairsnew)
    {
    ginfo->pair[npairsnew]=ginfo->pair[i];
    ginfo->pair[npairsnew].r1.ofs=ofs;
    ofs+=ginfo->pair[npairsnew].r1.nph;
    ginfo->pair[npairsnew].r2.ofs=ofs;
    if (ginfo->pair[npairsnew].r2.nph>0) ofs+=ginfo->pair[npairsnew].r2.nph;
    }
  ginfo->npairs=npairsnew;
  }
#endif

static ptrdiff_t get_nalms(const sharp_alm_info *ainfo)
  {
  ptrdiff_t res=0;
  for (int i=0; i<ainfo->nm; ++i)
    res += ainfo->lmax-ainfo->mval[i]+1;
  return res;
  }

static ptrdiff_t get_npix(const sharp_geom_info *ginfo)
  {
  ptrdiff_t res=0;
  for (int i=0; i<ginfo->npairs; ++i)
    {
    res += ginfo->pair[i].r1.nph;
    if (ginfo->pair[i].r2.nph>0) res += ginfo->pair[i].r2.nph;
    }
  return res;
  }

static double *get_sqsum_and_invert (dcmplx **alm, ptrdiff_t nalms, int ncomp)
  {
  double *sqsum=RALLOC(double,ncomp);
  for (int i=0; i<ncomp; ++i)
    {
    sqsum[i]=0;
    for (ptrdiff_t j=0; j<nalms; ++j)
      {
      sqsum[i]+=creal(alm[i][j])*creal(alm[i][j])
               +cimag(alm[i][j])*cimag(alm[i][j]);
      alm[i][j]=-alm[i][j];
      }
    }
  return sqsum;
  }

static void get_errors (dcmplx **alm, ptrdiff_t nalms, int ncomp, double *sqsum,
  double **err_abs, double **err_rel)
  {
  long nalms_tot=nalms;
#ifdef USE_MPI
  MPI_Allreduce(&nalms,&nalms_tot,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
#endif

  *err_abs=RALLOC(double,ncomp);
  *err_rel=RALLOC(double,ncomp);
  for (int i=0; i<ncomp; ++i)
    {
    double sum=0, maxdiff=0, sumtot, sqsumtot, maxdifftot;
    for (ptrdiff_t j=0; j<nalms; ++j)
      {
      double sqr=creal(alm[i][j])*creal(alm[i][j])
                +cimag(alm[i][j])*cimag(alm[i][j]);
      sum+=sqr;
      if (sqr>maxdiff) maxdiff=sqr;
      }
   maxdiff=sqrt(maxdiff);

#ifdef USE_MPI
    MPI_Allreduce(&sum,&sumtot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&sqsum[i],&sqsumtot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&maxdiff,&maxdifftot,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#else
    sumtot=sum;
    sqsumtot=sqsum[i];
    maxdifftot=maxdiff;
#endif
    sumtot=sqrt(sumtot/nalms_tot);
    sqsumtot=sqrt(sqsumtot/nalms_tot);
    (*err_abs)[i]=maxdifftot;
    (*err_rel)[i]=sumtot/sqsumtot;
    }
  }

static int good_fft_size(int n)
  {
  if (n<=6) return n;
  int bestfac=2*n;

  for (int f2=1; f2<bestfac; f2*=2)
    for (int f23=f2; f23<bestfac; f23*=3)
      for (int f235=f23; f235<bestfac; f235*=5)
        if (f235>=n) bestfac=f235;

  return bestfac;
  }

static void get_infos (const char *gname, int lmax, int *mmax, int *gpar1,
  int *gpar2, sharp_geom_info **ginfo, sharp_alm_info **ainfo)
  {
  UTIL_ASSERT(lmax>=0,"lmax must not be negative");
  if (*mmax<0) *mmax=lmax;
  UTIL_ASSERT(*mmax<=lmax,"mmax larger than lmax");

  if (mytask==0) printf ("lmax: %d, mmax: %d\n",lmax,*mmax);

  sharp_make_triangular_alm_info(lmax,*mmax,1,ainfo);
#ifdef USE_MPI
  reduce_alm_info(*ainfo);
#endif

  if (strcmp(gname,"healpix")==0)
    {
    if (*gpar1<1) *gpar1=lmax/2;
    if (*gpar1==0) ++(*gpar1);
    sharp_make_healpix_geom_info (*gpar1, 1, ginfo);
    if (mytask==0) printf ("HEALPix grid, nside=%d\n",*gpar1);
    }
  else if (strcmp(gname,"gauss")==0)
    {
    if (*gpar1<1) *gpar1=lmax+1;
    if (*gpar2<1) *gpar2=2*(*mmax)+1;
    sharp_make_gauss_geom_info (*gpar1, *gpar2, 0., 1, *gpar2, ginfo);
    if (mytask==0)
      printf ("Gauss-Legendre grid, nlat=%d, nlon=%d\n",*gpar1,*gpar2);
    }
  else if (strcmp(gname,"fejer1")==0)
    {
    if (*gpar1<1) *gpar1=2*lmax+1;
    if (*gpar2<1) *gpar2=2*(*mmax)+1;
    sharp_make_fejer1_geom_info (*gpar1, *gpar2, 0., 1, *gpar2, ginfo);
    if (mytask==0) printf ("Fejer1 grid, nlat=%d, nlon=%d\n",*gpar1,*gpar2);
    }
  else if (strcmp(gname,"fejer2")==0)
    {
    if (*gpar1<1) *gpar1=2*lmax+1;
    if (*gpar2<1) *gpar2=2*(*mmax)+1;
    sharp_make_fejer2_geom_info (*gpar1, *gpar2, 0., 1, *gpar2, ginfo);
    if (mytask==0) printf ("Fejer2 grid, nlat=%d, nlon=%d\n",*gpar1,*gpar2);
    }
  else if (strcmp(gname,"cc")==0)
    {
    if (*gpar1<1) *gpar1=2*lmax+1;
    if (*gpar2<1) *gpar2=2*(*mmax)+1;
    sharp_make_cc_geom_info (*gpar1, *gpar2, 0., 1, *gpar2, ginfo);
    if (mytask==0)
      printf("Clenshaw-Curtis grid, nlat=%d, nlon=%d\n",*gpar1,*gpar2);
    }
  else if (strcmp(gname,"smallgauss")==0)
    {
    int nlat=*gpar1, nlon=*gpar2;
    if (nlat<1) nlat=lmax+1;
    if (nlon<1) nlon=2*(*mmax)+1;
    *gpar1=nlat; *gpar2=nlon;
    sharp_make_gauss_geom_info (nlat, nlon, 0., 1, nlon, ginfo);
    ptrdiff_t npix_o=get_npix(*ginfo);
    size_t ofs=0;
    for (int i=0; i<(*ginfo)->npairs; ++i)
      {
      sharp_ringpair *pair=&((*ginfo)->pair[i]);
      int pring=1+2*sharp_get_mlim(lmax,0,pair->r1.sth,pair->r1.cth);
      if (pring>nlon) pring=nlon;
      pring=good_fft_size(pring);
      pair->r1.nph=pring;
      pair->r1.weight*=nlon*1./pring;
      pair->r1.ofs=ofs;
      ofs+=pring;
      if (pair->r2.nph>0)
        {
        pair->r2.nph=pring;
        pair->r2.weight*=nlon*1./pring;
        pair->r2.ofs=ofs;
        ofs+=pring;
        }
      }
    if (mytask==0)
      {
      ptrdiff_t npix=get_npix(*ginfo);
      printf("Small Gauss grid, nlat=%d, npix=%ld, savings=%.2f%%\n",
        nlat,(long)npix,(npix_o-npix)*100./npix_o);
      }
    }
  else
    UTIL_FAIL("unknown grid geometry");

#ifdef USE_MPI
  reduce_geom_info(*ginfo);
#endif
  }

static void check_sign_scale(void)
  {
  int lmax=50;
  int mmax=lmax;
  sharp_geom_info *tinfo;
  int nrings=lmax+1;
  int ppring=2*lmax+2;
  ptrdiff_t npix=(ptrdiff_t)nrings*ppring;
  sharp_make_gauss_geom_info (nrings, ppring, 0., 1, ppring, &tinfo);

  /* flip theta to emulate the "old" Gaussian grid geometry */
  for (int i=0; i<tinfo->npairs; ++i)
    {
    const double pi=3.141592653589793238462643383279502884197;
    tinfo->pair[i].r1.cth=-tinfo->pair[i].r1.cth;
    tinfo->pair[i].r2.cth=-tinfo->pair[i].r2.cth;
    tinfo->pair[i].r1.theta=pi-tinfo->pair[i].r1.theta;
    tinfo->pair[i].r2.theta=pi-tinfo->pair[i].r2.theta;
    }

  sharp_alm_info *alms;
  sharp_make_triangular_alm_info(lmax,mmax,1,&alms);
  ptrdiff_t nalms = ((mmax+1)*(mmax+2))/2 + (mmax+1)*(lmax-mmax);

  for (int ntrans=1; ntrans<10; ++ntrans)
    {
    double **map;
    ALLOC2D(map,double,2*ntrans,npix);

    dcmplx **alm;
    ALLOC2D(alm,dcmplx,2*ntrans,nalms);
    for (int i=0; i<2*ntrans; ++i)
      for (int j=0; j<nalms; ++j)
        alm[i][j]=1.+_Complex_I;

    sharp_execute(SHARP_ALM2MAP,0,&alm[0],&map[0],tinfo,alms,ntrans,SHARP_DP,
      NULL,NULL);
    for (int it=0; it<ntrans; ++it)
      {
      UTIL_ASSERT(FAPPROX(map[it][0     ], 3.588246976618616912e+00,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[it][npix/2], 4.042209792157496651e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[it][npix-1],-1.234675107554816442e+01,1e-12),
        "error");
      }
    sharp_execute(SHARP_ALM2MAP,1,&alm[0],&map[0],tinfo,alms,ntrans,SHARP_DP,
      NULL,NULL);
    for (int it=0; it<ntrans; ++it)
      {
      UTIL_ASSERT(FAPPROX(map[2*it  ][0     ], 2.750897760535633285e+00,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix/2], 3.137704477368562905e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix-1],-8.405730859837063917e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][0     ],-2.398026536095463346e+00,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix/2],-4.961140548331700728e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix-1],-1.412765834230440021e+01,1e-12),
        "error");
      }

    sharp_execute(SHARP_ALM2MAP,2,&alm[0],&map[0],tinfo,alms,ntrans,SHARP_DP,
      NULL,NULL);
    for (int it=0; it<ntrans; ++it)
      {
      UTIL_ASSERT(FAPPROX(map[2*it  ][0     ],-1.398186224727334448e+00,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix/2],-2.456676000884031197e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix-1],-1.516249174408820863e+02,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][0     ],-3.173406200299964119e+00,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix/2],-5.831327404513146462e+01,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix-1],-1.863257892248353897e+01,1e-12),
        "error");
      }

    sharp_execute(SHARP_ALM2MAP_DERIV1,1,&alm[0],&map[0],tinfo,alms,ntrans,
      SHARP_DP,NULL,NULL);
    for (int it=0; it<ntrans; ++it)
      {
      UTIL_ASSERT(FAPPROX(map[2*it  ][0     ],-6.859393905369091105e-01,1e-11),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix/2],-2.103947835973212364e+02,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it  ][npix-1],-1.092463246472086439e+03,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][0     ],-1.411433220713928165e+02,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix/2],-1.146122859381925082e+03,1e-12),
        "error");
      UTIL_ASSERT(FAPPROX(map[2*it+1][npix-1], 7.821618677689795049e+02,1e-12),
        "error");
      }

    DEALLOC2D(map);
    DEALLOC2D(alm);
    }

  sharp_destroy_alm_info(alms);
  sharp_destroy_geom_info(tinfo);
  }

static void do_sht (sharp_geom_info *ginfo, sharp_alm_info *ainfo,
  int spin, int ntrans, int nv, double **err_abs, double **err_rel,
  double *t_a2m, double *t_m2a, unsigned long long *op_a2m,
  unsigned long long *op_m2a)
  {
  ptrdiff_t nalms = get_nalms(ainfo);
  int ncomp = ntrans*((spin==0) ? 1 : 2);

  size_t npix = get_npix(ginfo);
  double **map;
  ALLOC2D(map,double,ncomp,npix);
  for (int i=0; i<ncomp; ++i)
    SET_ARRAY(map[i],0,(int)npix,0);

  dcmplx **alm;
  ALLOC2D(alm,dcmplx,ncomp,nalms);
  for (int i=0; i<ncomp; ++i)
    random_alm(alm[i],ainfo,spin,i+1);

#ifdef USE_MPI
  sharp_execute_mpi(MPI_COMM_WORLD,SHARP_ALM2MAP,spin,&alm[0],&map[0],ginfo,
    ainfo,ntrans, SHARP_DP|SHARP_ADD|nv,t_a2m,op_a2m);
#else
  sharp_execute(SHARP_ALM2MAP,spin,&alm[0],&map[0],ginfo,ainfo,ntrans,
    SHARP_DP|nv,t_a2m,op_a2m);
#endif
  if (t_a2m!=NULL) *t_a2m=maxTime(*t_a2m);
  if (op_a2m!=NULL) *op_a2m=totalops(*op_a2m);
  double *sqsum=get_sqsum_and_invert(alm,nalms,ncomp);
#ifdef USE_MPI
  sharp_execute_mpi(MPI_COMM_WORLD,SHARP_MAP2ALM,spin,&alm[0],&map[0],ginfo,
    ainfo,ntrans,SHARP_DP|SHARP_ADD|nv,t_m2a,op_m2a);
#else
  sharp_execute(SHARP_MAP2ALM,spin,&alm[0],&map[0],ginfo,ainfo,ntrans,
    SHARP_DP|SHARP_ADD|nv,t_m2a,op_m2a);
#endif
  if (t_m2a!=NULL) *t_m2a=maxTime(*t_m2a);
  if (op_m2a!=NULL) *op_m2a=totalops(*op_m2a);
  get_errors(alm, nalms, ncomp, sqsum, err_abs, err_rel);

  DEALLOC(sqsum);
  DEALLOC2D(map);
  DEALLOC2D(alm);
  }

static void check_accuracy (sharp_geom_info *ginfo, sharp_alm_info *ainfo,
  int spin, int ntrans, int nv)
  {
  int ncomp = ntrans*((spin==0) ? 1 : 2);
  double *err_abs, *err_rel;
  do_sht (ginfo, ainfo, spin, ntrans, nv, &err_abs, &err_rel, NULL, NULL,
    NULL, NULL);
  for (int i=0; i<ncomp; ++i)
    UTIL_ASSERT((err_rel[i]<1e-10) && (err_abs[i]<1e-10),"error");
  DEALLOC(err_rel);
  DEALLOC(err_abs);
  }

static void sharp_acctest(void)
  {
  if (mytask==0) sharp_module_startup("sharp_acctest",1,1,"",1);

  if (mytask==0) printf("Checking signs and scales.\n");
  check_sign_scale();
  if (mytask==0) printf("Passed.\n\n");

  if (mytask==0) printf("Testing map analysis accuracy.\n");

  sharp_geom_info *ginfo;
  sharp_alm_info *ainfo;
  int lmax=127, mmax=127, nlat=128, nlon=256;
  get_infos ("gauss", lmax, &mmax, &nlat, &nlon, &ginfo, &ainfo);
  for (int nv=1; nv<=6; ++nv)
    for (int ntrans=1; ntrans<=6; ++ntrans)
      {
      check_accuracy(ginfo,ainfo,0,ntrans,nv);
      check_accuracy(ginfo,ainfo,1,ntrans,nv);
      check_accuracy(ginfo,ainfo,2,ntrans,nv);
      check_accuracy(ginfo,ainfo,3,ntrans,nv);
      check_accuracy(ginfo,ainfo,30,ntrans,nv);
      }
  sharp_destroy_alm_info(ainfo);
  sharp_destroy_geom_info(ginfo);
  if (mytask==0) printf("Passed.\n\n");
  }

static void sharp_test (int argc, const char **argv)
  {
  if (mytask==0) sharp_announce("sharp_test");
  UTIL_ASSERT(argc>=9,"usage: grid lmax mmax geom1 geom2 spin ntrans");
  int lmax=atoi(argv[3]);
  int mmax=atoi(argv[4]);
  int gpar1=atoi(argv[5]);
  int gpar2=atoi(argv[6]);
  int spin=atoi(argv[7]);
  int ntrans=atoi(argv[8]);

  if (mytask==0) printf("Testing map analysis accuracy.\n");
  if (mytask==0) printf("spin=%d, ntrans=%d\n", spin, ntrans);

  sharp_geom_info *ginfo;
  sharp_alm_info *ainfo;
  get_infos (argv[2], lmax, &mmax, &gpar1, &gpar2, &ginfo, &ainfo);

  int ncomp = ntrans*((spin==0) ? 1 : 2);
  double t_a2m=1e30, t_m2a=1e30;
  unsigned long long op_a2m, op_m2a;
  double *err_abs,*err_rel;

  double t_acc=0;
  int nrpt=0;
  while(1)
    {
    ++nrpt;
    double ta2m2, tm2a2;
    do_sht (ginfo, ainfo, spin, ntrans, 0, &err_abs, &err_rel, &ta2m2, &tm2a2,
      &op_a2m, &op_m2a);
    if (ta2m2<t_a2m) t_a2m=ta2m2;
    if (tm2a2<t_m2a) t_m2a=tm2a2;
    t_acc+=t_a2m+t_m2a;
    if (t_acc>2.)
      {
      if (mytask==0) printf("Best of %d runs\n",nrpt);
      break;
      }
    DEALLOC(err_abs);
    DEALLOC(err_rel);
    }

  if (mytask==0) printf("wall time for alm2map: %fs\n",t_a2m);
  if (mytask==0) printf("Performance: %fGFLOPs/s\n",1e-9*op_a2m/t_a2m);
  if (mytask==0) printf("wall time for map2alm: %fs\n",t_m2a);
  if (mytask==0) printf("Performance: %fGFLOPs/s\n",1e-9*op_m2a/t_m2a);

  if (mytask==0)
    for (int i=0; i<ncomp; ++i)
      printf("component %i: rms %e, maxerr %e\n",i,err_rel[i], err_abs[i]);

  double iosize = ncomp*(16.*get_nalms(ainfo) + 8.*get_npix(ginfo));
  iosize = allreduceSumDouble(iosize);

  sharp_destroy_alm_info(ainfo);
  sharp_destroy_geom_info(ginfo);

  double tmem=totalMem();
  if (mytask==0)
    printf("\nMemory high water mark: %.2f MB\n",tmem/(1<<20));
  if (mytask==0)
    printf("Memory overhead: %.2f MB (%.2f%% of working set)\n",
      (tmem-iosize)/(1<<20),100.*(1.-iosize/tmem));

#ifdef _OPENMP
  int nomp=omp_get_max_threads();
#else
  int nomp=1;
#endif

  double maxerel=0., maxeabs=0.;
  for (int i=0; i<ncomp; ++i)
    {
    if (maxerel<err_rel[i]) maxerel=err_rel[i];
    if (maxeabs<err_abs[i]) maxeabs=err_abs[i];
    }

  if (mytask==0)
    printf("%-12s %-10s %2d %d %2d %3d %6d %6d %6d %6d %2d %.2e %7.2f %.2e %7.2f"
           " %9.2f %6.2f %.2e %.2e\n",
      getenv("HOST"),argv[2],spin,VLEN,nomp,ntasks,lmax,mmax,gpar1,gpar2,
      ntrans,t_a2m,1e-9*op_a2m/t_a2m,t_m2a,1e-9*op_m2a/t_m2a,tmem/(1<<20),
      100.*(1.-iosize/tmem),maxerel,maxeabs);

  DEALLOC(err_abs);
  DEALLOC(err_rel);
  }

static void sharp_bench (int argc, const char **argv)
  {
  if (mytask==0) sharp_announce("sharp_bench");
  UTIL_ASSERT(argc>=9,"usage: grid lmax mmax geom1 geom2 spin ntrans");
  int lmax=atoi(argv[3]);
  int mmax=atoi(argv[4]);
  int gpar1=atoi(argv[5]);
  int gpar2=atoi(argv[6]);
  int spin=atoi(argv[7]);
  int ntrans=atoi(argv[8]);

  if (mytask==0) printf("Testing map analysis accuracy.\n");
  if (mytask==0) printf("spin=%d, ntrans=%d\n", spin, ntrans);

  sharp_geom_info *ginfo;
  sharp_alm_info *ainfo;
  get_infos (argv[2], lmax, &mmax, &gpar1, &gpar2, &ginfo, &ainfo);

  double ta2m_auto=1e30, tm2a_auto=1e30, ta2m_min=1e30, tm2a_min=1e30;
  unsigned long long opa2m_min=0, opm2a_min=0;
  int nvmin_a2m=-1, nvmin_m2a=-1;
  for (int nv=0; nv<=6; ++nv)
    {
    int ntries=0;
    double tacc=0;
    do
      {
      double t_a2m, t_m2a;
      unsigned long long op_a2m, op_m2a;
      double *err_abs,*err_rel;
      do_sht (ginfo, ainfo, spin, ntrans, nv, &err_abs, &err_rel,
        &t_a2m, &t_m2a, &op_a2m, &op_m2a);

      DEALLOC(err_abs);
      DEALLOC(err_rel);
      tacc+=t_a2m+t_m2a;
      ++ntries;
      if (nv==0)
        {
        if (t_a2m<ta2m_auto) ta2m_auto=t_a2m;
        if (t_m2a<tm2a_auto) tm2a_auto=t_m2a;
        }
      else
        {
        if (t_a2m<ta2m_min) { nvmin_a2m=nv; ta2m_min=t_a2m; opa2m_min=op_a2m; }
        if (t_m2a<tm2a_min) { nvmin_m2a=nv; tm2a_min=t_m2a; opm2a_min=op_m2a; }
        }
      } while((ntries<2)||(tacc<3.));
    }
  if (mytask==0)
    {
    printf("a2m: nvmin=%d tmin=%fs speedup=%.2f%% perf=%.2fGFlops/s\n",
      nvmin_a2m,ta2m_min,100.*(ta2m_auto-ta2m_min)/ta2m_auto,
      1e-9*opa2m_min/ta2m_min);
    printf("m2a: nvmin=%d tmin=%fs speedup=%.2f%% perf=%.2fGFlops/s\n",
      nvmin_m2a,tm2a_min,100.*(tm2a_auto-tm2a_min)/tm2a_auto,
      1e-9*opm2a_min/tm2a_min);
    }

  sharp_destroy_alm_info(ainfo);
  sharp_destroy_geom_info(ginfo);
  }

int main(int argc, const char **argv)
  {
#ifdef USE_MPI
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytask);
#else
  mytask=0; ntasks=1;
#endif

  UTIL_ASSERT(argc>=2,"need at least one command line argument");

  if (strcmp(argv[1],"acctest")==0)
    sharp_acctest();
  else if (strcmp(argv[1],"test")==0)
    sharp_test(argc,argv);
  else if (strcmp(argv[1],"bench")==0)
    sharp_bench(argc,argv);
  else
    UTIL_FAIL("unknown command");

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
  }
