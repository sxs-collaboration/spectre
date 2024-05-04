/* Function adapted from GNU GSL file glfixed.c
   Original author: Pavel Holoborodko (http://www.holoborodko.com)

   Adjustments by M. Reinecke
    - adjusted interface (keep epsilon internal, return full number of points)
    - removed precomputed tables
    - tweaked Newton iteration to obtain higher accuracy

   Changes by SXS Collaboration:
   1. #pragma omp lines are commented out.
 */

#include <math.h>
#include "sharp_legendre_roots.h"
#include "c_utils.h"

static inline double one_minus_x2 (double x)
  { return (fabs(x)>0.1) ? (1.+x)*(1.-x) : 1.-x*x; }

void sharp_legendre_roots(int n, double *x, double *w)
  {
  const double pi = 3.141592653589793238462643383279502884197;
  const double eps = 3e-14;
  int m = (n+1)>>1;

  double t0 = 1 - (1-1./n) / (8.*n*n);
  double t1 = 1./(4.*n+2.);

/* #pragma omp parallel */
{
  int i;
/* #pragma omp for schedule(dynamic,100) */
  for (i=1; i<=m; ++i)
    {
    double x0 = cos(pi * ((i<<2)-1) * t1) * t0;

    int dobreak=0;
    int j=0;
    double dpdx;
    while(1)
      {
      double P_1 = 1.0;
      double P0 = x0;
      double dx, x1;

      for (int k=2; k<=n; k++)
        {
        double P_2 = P_1;
        P_1 = P0;
//        P0 = ((2*k-1)*x0*P_1-(k-1)*P_2)/k;
        P0 = x0*P_1 + (k-1.)/k * (x0*P_1-P_2);
        }

      dpdx = (P_1 - x0*P0) * n / one_minus_x2(x0);

      /* Newton step */
      x1 = x0 - P0/dpdx;
      dx = x0-x1;
      x0 = x1;
      if (dobreak) break;

      if (fabs(dx)<=eps) dobreak=1;
      UTIL_ASSERT(++j<100,"convergence problem");
      }

    x[i-1] = -x0;
    x[n-i] = x0;
    w[i-1] = w[n-i] = 2. / (one_minus_x2(x0) * dpdx * dpdx);
    }
} // end of parallel region
  }
