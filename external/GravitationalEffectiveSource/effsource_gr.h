/*******************************************************************************
 * Copyright (C) 2011 Barry Wardell
 ******************************************************************************/

struct coordinate {
  double t;
  double r;
  double theta;
  double phi;
};

void effsource_init(double M, double a);
void effsource_set_particle(struct coordinate * x, double E, double L, double ur);

void effsource_hS(struct coordinate * x, double * hS);
void effsource_hS_m(int m, struct coordinate * x, double * hS_re, double * hS_im);

void effsource_calc(struct coordinate * x, double *hS, double *dhS_dr, double *dhS_dth,
                    double *dhS_dph, double *dhS_dt, double *src);
void effsource_calc_m(int m, struct coordinate * x, double *hS_re, double *hS_im,
   double *dhS_dr_re, double *dhS_dr_im, double *dhS_dth_re, double *dhS_dth_im,
   double *dhS_dph_re, double *dhS_dph_im, double *dhS_dt_re, double *dhS_dt_im,
   double *src_re, double *src_im);
