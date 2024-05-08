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

/*! \file sharp_mpi.c
 *  Functionality only needed for MPI-parallel transforms
 *
 *  Copyright (C) 2012-2013 Max-Planck-Society
 *  \author Martin Reinecke \author Dag Sverre Seljebotn
 */

#ifdef USE_MPI

#include "sharp_mpi.h"

typedef struct
  {
  int ntasks;     /* number of tasks */
  int mytask;     /* own task number */
  MPI_Comm comm;  /* communicator to use */

  int *nm;        /* number of m values on every task */
  int *ofs_m;     /* accumulated nm */
  int nmtotal;    /* total number of m values (must be mmax+1) */
  int *mval;      /* array containing all m values of task 0, task 1 etc. */
  int mmax;
  int nph;

  int *npair;     /* number of ring pairs on every task */
  int *ofs_pair;  /* accumulated npair */
  int npairtotal; /* total number of ring pairs */

  double *theta;  /* theta of first ring of every pair on task 0, task 1 etc. */
  int *ispair;    /* is this really a pair? */

  int *almcount, *almdisp, *mapcount, *mapdisp; /* for all2all communication */
  } sharp_mpi_info;

static void sharp_make_mpi_info (MPI_Comm comm, const sharp_job *job,
  sharp_mpi_info *minfo)
  {
  minfo->comm = comm;
  MPI_Comm_size (comm, &minfo->ntasks);
  MPI_Comm_rank (comm, &minfo->mytask);

  minfo->nm=RALLOC(int,minfo->ntasks);
  MPI_Allgather ((int *)(&job->ainfo->nm),1,MPI_INT,minfo->nm,1,MPI_INT,comm);
  minfo->ofs_m=RALLOC(int,minfo->ntasks+1);
  minfo->ofs_m[0]=0;
  for (int i=1; i<=minfo->ntasks; ++i)
    minfo->ofs_m[i] = minfo->ofs_m[i-1]+minfo->nm[i-1];
  minfo->nmtotal=minfo->ofs_m[minfo->ntasks];
  minfo->mval=RALLOC(int,minfo->nmtotal);
  MPI_Allgatherv(job->ainfo->mval, job->ainfo->nm, MPI_INT, minfo->mval,
    minfo->nm, minfo->ofs_m, MPI_INT, comm);

  minfo->mmax=sharp_get_mmax(minfo->mval,minfo->nmtotal);

  minfo->npair=RALLOC(int,minfo->ntasks);
  MPI_Allgather ((int *)(&job->ginfo->npairs), 1, MPI_INT, minfo->npair, 1,
    MPI_INT, comm);
  minfo->ofs_pair=RALLOC(int,minfo->ntasks+1);
  minfo->ofs_pair[0]=0;
  for (int i=1; i<=minfo->ntasks; ++i)
    minfo->ofs_pair[i] = minfo->ofs_pair[i-1]+minfo->npair[i-1];
  minfo->npairtotal=minfo->ofs_pair[minfo->ntasks];

  double *theta_tmp=RALLOC(double,job->ginfo->npairs);
  int *ispair_tmp=RALLOC(int,job->ginfo->npairs);
  for (int i=0; i<job->ginfo->npairs; ++i)
    {
    theta_tmp[i]=job->ginfo->pair[i].r1.theta;
    ispair_tmp[i]=job->ginfo->pair[i].r2.nph>0;
    }
  minfo->theta=RALLOC(double,minfo->npairtotal);
  minfo->ispair=RALLOC(int,minfo->npairtotal);
  MPI_Allgatherv(theta_tmp, job->ginfo->npairs, MPI_DOUBLE, minfo->theta,
    minfo->npair, minfo->ofs_pair, MPI_DOUBLE, comm);
  MPI_Allgatherv(ispair_tmp, job->ginfo->npairs, MPI_INT, minfo->ispair,
    minfo->npair, minfo->ofs_pair, MPI_INT, comm);
  DEALLOC(theta_tmp);
  DEALLOC(ispair_tmp);

  minfo->nph=2*job->nmaps*job->ntrans;

  minfo->almcount=RALLOC(int,minfo->ntasks);
  minfo->almdisp=RALLOC(int,minfo->ntasks+1);
  minfo->mapcount=RALLOC(int,minfo->ntasks);
  minfo->mapdisp=RALLOC(int,minfo->ntasks+1);
  minfo->almdisp[0]=minfo->mapdisp[0]=0;
  for (int i=0; i<minfo->ntasks; ++i)
    {
    minfo->almcount[i] = 2*minfo->nph*minfo->nm[minfo->mytask]*minfo->npair[i];
    minfo->almdisp[i+1] = minfo->almdisp[i]+minfo->almcount[i];
    minfo->mapcount[i] = 2*minfo->nph*minfo->nm[i]*minfo->npair[minfo->mytask];
    minfo->mapdisp[i+1] = minfo->mapdisp[i]+minfo->mapcount[i];
    }
  }

static void sharp_destroy_mpi_info (sharp_mpi_info *minfo)
  {
  DEALLOC(minfo->nm);
  DEALLOC(minfo->ofs_m);
  DEALLOC(minfo->mval);
  DEALLOC(minfo->npair);
  DEALLOC(minfo->ofs_pair);
  DEALLOC(minfo->theta);
  DEALLOC(minfo->ispair);
  DEALLOC(minfo->almcount);
  DEALLOC(minfo->almdisp);
  DEALLOC(minfo->mapcount);
  DEALLOC(minfo->mapdisp);
  }

static void sharp_communicate_alm2map (const sharp_mpi_info *minfo, dcmplx **ph)
  {
  dcmplx *phas_tmp = RALLOC(dcmplx,minfo->mapdisp[minfo->ntasks]/2);

  MPI_Alltoallv (*ph,minfo->almcount,minfo->almdisp,MPI_DOUBLE,phas_tmp,
    minfo->mapcount,minfo->mapdisp,MPI_DOUBLE,minfo->comm);

  DEALLOC(*ph);
  ALLOC(*ph,dcmplx,minfo->nph*minfo->npair[minfo->mytask]*minfo->nmtotal);

  for (int task=0; task<minfo->ntasks; ++task)
    for (int th=0; th<minfo->npair[minfo->mytask]; ++th)
      for (int mi=0; mi<minfo->nm[task]; ++mi)
        {
        int m = minfo->mval[mi+minfo->ofs_m[task]];
        int o1 = minfo->nph*(th*(minfo->mmax+1) + m);
        int o2 = minfo->mapdisp[task]/2+minfo->nph*(mi+th*minfo->nm[task]);
        for (int i=0; i<minfo->nph; ++i)
          (*ph)[o1+i] = phas_tmp[o2+i];
        }
  DEALLOC(phas_tmp);
  }

static void sharp_communicate_map2alm (const sharp_mpi_info *minfo, dcmplx **ph)
  {
  dcmplx *phas_tmp = RALLOC(dcmplx,minfo->mapdisp[minfo->ntasks]/2);

  for (int task=0; task<minfo->ntasks; ++task)
    for (int th=0; th<minfo->npair[minfo->mytask]; ++th)
      for (int mi=0; mi<minfo->nm[task]; ++mi)
        {
        int m = minfo->mval[mi+minfo->ofs_m[task]];
        int o1 = minfo->mapdisp[task]/2+minfo->nph*(mi+th*minfo->nm[task]);
        int o2 = minfo->nph*(th*(minfo->mmax+1) + m);
        for (int i=0; i<minfo->nph; ++i)
          phas_tmp[o1+i] = (*ph)[o2+i];
        }

  DEALLOC(*ph);
  ALLOC(*ph,dcmplx,minfo->nph*minfo->nm[minfo->mytask]*minfo->npairtotal);

  MPI_Alltoallv (phas_tmp,minfo->mapcount,minfo->mapdisp,MPI_DOUBLE,
    *ph,minfo->almcount,minfo->almdisp,MPI_DOUBLE,minfo->comm);

  DEALLOC(phas_tmp);
  }

static void alloc_phase_mpi (sharp_job *job, int nm, int ntheta,
  int nmfull, int nthetafull)
  {
  ptrdiff_t phase_size = (job->type==SHARP_MAP2ALM) ?
    (ptrdiff_t)(nmfull)*ntheta : (ptrdiff_t)(nm)*nthetafull;
  job->phase=RALLOC(dcmplx,2*job->ntrans*job->nmaps*phase_size);
  job->s_m=2*job->ntrans*job->nmaps;
  job->s_th = job->s_m * ((job->type==SHARP_MAP2ALM) ? nmfull : nm);
  }

static void alm2map_comm (sharp_job *job, const sharp_mpi_info *minfo)
  {
  if (job->type != SHARP_MAP2ALM)
    {
    sharp_communicate_alm2map (minfo,&job->phase);
    job->s_th=job->s_m*minfo->nmtotal;
    }
  }

static void map2alm_comm (sharp_job *job, const sharp_mpi_info *minfo)
  {
  if (job->type == SHARP_MAP2ALM)
    {
    sharp_communicate_map2alm (minfo,&job->phase);
    job->s_th=job->s_m*minfo->nm[minfo->mytask];
    }
  }

static void sharp_execute_job_mpi (sharp_job *job, MPI_Comm comm)
  {
  int ntasks;
  MPI_Comm_size(comm, &ntasks);
  if (ntasks==1) /* fall back to scalar implementation */
    { sharp_execute_job (job); return; }

  MPI_Barrier(comm);
  double timer=wallTime();
  job->opcnt=0;
  sharp_mpi_info minfo;
  sharp_make_mpi_info(comm, job, &minfo);

  if (minfo.npairtotal>minfo.ntasks*300)
    {
    int nsub=(minfo.npairtotal+minfo.ntasks*200-1)/(minfo.ntasks*200);
    for (int isub=0; isub<nsub; ++isub)
      {
      sharp_job ljob=*job;
      // When creating a_lm, every sub-job produces a complete set of
      // coefficients; they need to be added up.
      if ((isub>0)&&(job->type==SHARP_MAP2ALM)) ljob.flags|=SHARP_ADD;
      sharp_geom_info lginfo;
      lginfo.pair=RALLOC(sharp_ringpair,(job->ginfo->npairs/nsub)+1);
      lginfo.npairs=0;
      lginfo.nphmax = job->ginfo->nphmax;
      while (lginfo.npairs*nsub+isub<job->ginfo->npairs)
        {
        lginfo.pair[lginfo.npairs]=job->ginfo->pair[lginfo.npairs*nsub+isub];
        ++lginfo.npairs;
        }
      ljob.ginfo=&lginfo;
      sharp_execute_job_mpi (&ljob,comm);
      job->opcnt+=ljob.opcnt;
      DEALLOC(lginfo.pair);
      }
    }
  else
    {
    int lmax = job->ainfo->lmax;
    job->norm_l = sharp_Ylmgen_get_norm (lmax, job->spin);

    /* clear output arrays if requested */
    init_output (job);

    alloc_phase_mpi (job,job->ainfo->nm,job->ginfo->npairs,minfo.mmax+1,
      minfo.npairtotal);

    double *cth = RALLOC(double,minfo.npairtotal),
          *sth = RALLOC(double,minfo.npairtotal);
    int *mlim = RALLOC(int,minfo.npairtotal);
    for (int i=0; i<minfo.npairtotal; ++i)
      {
      cth[i] = cos(minfo.theta[i]);
      sth[i] = sin(minfo.theta[i]);
      mlim[i] = sharp_get_mlim(lmax, job->spin, sth[i], cth[i]);
      }

    /* map->phase where necessary */
    map2phase (job, minfo.mmax, 0, job->ginfo->npairs);

    map2alm_comm (job, &minfo);

#pragma omp parallel if ((job->flags&SHARP_NO_OPENMP)==0)
{
    sharp_job ljob = *job;
    sharp_Ylmgen_C generator;
    sharp_Ylmgen_init (&generator,lmax,minfo.mmax,ljob.spin);
    alloc_almtmp(&ljob,lmax);

#pragma omp for schedule(dynamic,1)
    for (int mi=0; mi<job->ainfo->nm; ++mi)
      {
  /* alm->alm_tmp where necessary */
      alm2almtmp (&ljob, lmax, mi);

  /* inner conversion loop */
      inner_loop (&ljob, minfo.ispair, cth, sth, 0, minfo.npairtotal,
        &generator, mi, mlim);

  /* alm_tmp->alm where necessary */
      almtmp2alm (&ljob, lmax, mi);
      }

    sharp_Ylmgen_destroy(&generator);
    dealloc_almtmp(&ljob);

#pragma omp critical
    job->opcnt+=ljob.opcnt;
} /* end of parallel region */

    alm2map_comm (job, &minfo);

  /* phase->map where necessary */
    phase2map (job, minfo.mmax, 0, job->ginfo->npairs);

    DEALLOC(mlim);
    DEALLOC(cth);
    DEALLOC(sth);
    DEALLOC(job->norm_l);
    dealloc_phase (job);
    }
  sharp_destroy_mpi_info(&minfo);
  job->time=wallTime()-timer;
  }

void sharp_execute_mpi (MPI_Comm comm, sharp_jobtype type, int spin,
  void *alm, void *map, const sharp_geom_info *geom_info,
  const sharp_alm_info *alm_info, int ntrans, int flags, double *time,
  unsigned long long *opcnt)
  {
  sharp_job job;
  sharp_build_job_common (&job, type, spin, alm, map, geom_info, alm_info,
    ntrans, flags);

  sharp_execute_job_mpi (&job, comm);
  if (time!=NULL) *time = job.time;
  if (opcnt!=NULL) *opcnt = job.opcnt;
  }

/* We declare this only in C file to make symbol available for Fortran wrappers;
   without declaring it in C header as it should not be available to C code */
void sharp_execute_mpi_fortran(MPI_Fint comm, sharp_jobtype type, int spin,
  void *alm, void *map, const sharp_geom_info *geom_info,
  const sharp_alm_info *alm_info, int ntrans, int flags, double *time,
  unsigned long long *opcnt);
void sharp_execute_mpi_fortran(MPI_Fint comm, sharp_jobtype type, int spin,
  void *alm, void *map, const sharp_geom_info *geom_info,
  const sharp_alm_info *alm_info, int ntrans, int flags, double *time,
  unsigned long long *opcnt)
  {
  sharp_execute_mpi(MPI_Comm_f2c(comm), type, spin, alm, map, geom_info,
                    alm_info, ntrans, flags, time, opcnt);
  }

#endif
