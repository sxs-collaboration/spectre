/*******************************************************************************
 * Copyright (C) 2011 Barry Wardell
 ******************************************************************************/

struct coordinate {
  double r;
  double theta;
  double phi;
  double t;
};

void effsource_init(double M, double a);
void effsource_set_particle(struct coordinate * x_p, double e, double l, double ur_p);

void effsource_PhiS(struct coordinate * x, double * PhiS);
void effsource_calc(struct coordinate * x,
  double * PhiS, double * dPhiS_dx, double * d2PhiS_dx2, double * src);

void effsource_PhiS_m(int m, struct coordinate * x, double * PhiS);
void effsource_calc_m(int m, struct coordinate * x,
  double * PhiS, double * dPhiS_dx, double * d2PhiS_dx2, double * src);
