// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbitConvertEffsource.hpp"

#include <cmath>

// NOLINTBEGIN
namespace GrSelfForce::detail {

void convert_effsource_psi(const int m, const double a, const double r,
                           const double th, std::array<double, 10>& real_orig,
                           std::array<double, 10>& imag_orig,
                           std::array<double, 10>& real_conv,
                           std::array<double, 10>& imag_conv) {
  const double factor = r / (2 * M_PI);
  const double rplus = 1 + sqrt(1 - a * a);
  const double rminus = 1 - sqrt(1 - a * a);
  const double mdphi =
      m * a / (rplus - rminus) * log((r - rplus) / (r - rminus));
  const double cosmdphi = cos(mdphi);
  const double sinmdphi = sin(mdphi);
  const double sinth = sin(th);
  real_conv[0] = factor * (real_orig[0] * cosmdphi + imag_orig[0] * sinmdphi);
  imag_conv[0] = factor * (imag_orig[0] * cosmdphi - real_orig[0] * sinmdphi);
  real_conv[1] = factor * (real_orig[1] * cosmdphi + imag_orig[1] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r);
  imag_conv[1] = factor * (imag_orig[1] * cosmdphi - real_orig[1] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r);
  real_conv[2] =
      factor * (real_orig[2] * cosmdphi + imag_orig[2] * sinmdphi) / r;
  imag_conv[2] =
      factor * (imag_orig[2] * cosmdphi - real_orig[2] * sinmdphi) / r;
  real_conv[3] = factor * (real_orig[3] * cosmdphi + imag_orig[3] * sinmdphi) /
                 (r * sinth);
  imag_conv[3] = factor * (imag_orig[3] * cosmdphi - real_orig[3] * sinmdphi) /
                 (r * sinth);
  real_conv[4] = factor * (real_orig[4] * cosmdphi + imag_orig[4] * sinmdphi) *
                 (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) /
                 (r * r * r * r);
  imag_conv[4] = factor * (imag_orig[4] * cosmdphi - real_orig[4] * sinmdphi) *
                 (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) /
                 (r * r * r * r);
  real_conv[5] = factor * (real_orig[5] * cosmdphi + imag_orig[5] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r * r);
  imag_conv[5] = factor * (imag_orig[5] * cosmdphi - real_orig[5] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r * r);
  real_conv[6] = factor * (real_orig[6] * cosmdphi + imag_orig[6] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r * r * sinth);
  imag_conv[6] = factor * (imag_orig[6] * cosmdphi - real_orig[6] * sinmdphi) *
                 (r * r - 2 * r + a * a) / (r * r * r * sinth);
  real_conv[7] =
      factor * (real_orig[7] * cosmdphi + imag_orig[7] * sinmdphi) / (r * r);
  imag_conv[7] =
      factor * (imag_orig[7] * cosmdphi - real_orig[7] * sinmdphi) / (r * r);
  real_conv[8] = factor * (real_orig[8] * cosmdphi + imag_orig[8] * sinmdphi) /
                 (r * r * sinth);
  imag_conv[8] = factor * (imag_orig[8] * cosmdphi - real_orig[8] * sinmdphi) /
                 (r * r * sinth);
  real_conv[9] = factor * (real_orig[9] * cosmdphi + imag_orig[9] * sinmdphi) /
                 (r * r * sinth * sinth);
  imag_conv[9] = factor * (imag_orig[9] * cosmdphi - real_orig[9] * sinmdphi) /
                 (r * r * sinth * sinth);
}

void convert_effsource_Seff(const int m, const double a, const double r,
                            const double th, std::array<double, 10>& real_orig,
                            std::array<double, 10>& imag_orig,
                            std::array<double, 10>& real_conv,
                            std::array<double, 10>& imag_conv) {
  const double factor = 1.0 / (2 * M_PI);
  const double rplus = 1 + sqrt(1 - a * a);
  const double rminus = 1 - sqrt(1 - a * a);
  const double mdphi =
      m * a / (rplus - rminus) * log((r - rplus) / (r - rminus));
  const double cosmdphi = cos(mdphi);
  const double sinmdphi = sin(mdphi);
  const double sinth = sin(th);
  const double costh = cos(th);
  const double cos_sq_th = costh * costh;
  const double cos_2th = cos(2 * th);
  real_conv[0] = factor * (real_orig[0] * cosmdphi + imag_orig[0] * sinmdphi) *
                 (-((r * (a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    ((a * a + r * r) * (a * a + r * r))));
  imag_conv[0] = factor * (imag_orig[0] * cosmdphi - real_orig[0] * sinmdphi) *
                 (-((r * (a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    ((a * a + r * r) * (a * a + r * r))));
  real_conv[1] = factor * (real_orig[1] * cosmdphi + imag_orig[1] * sinmdphi) *
                 (-0.5 *
                  ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                   (a * a + 2 * r * r + a * a * cos_2th)) /
                  (r * (a * a + r * r) * (a * a + r * r)));
  imag_conv[1] = factor * (imag_orig[1] * cosmdphi - real_orig[1] * sinmdphi) *
                 (-0.5 *
                  ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                   (a * a + 2 * r * r + a * a * cos_2th)) /
                  (r * (a * a + r * r) * (a * a + r * r)));
  real_conv[2] = factor * (real_orig[2] * cosmdphi + imag_orig[2] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    ((a * a + r * r) * (a * a + r * r))));
  imag_conv[2] = factor * (imag_orig[2] * cosmdphi - real_orig[2] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    ((a * a + r * r) * (a * a + r * r))));
  real_conv[3] = factor * (real_orig[3] * cosmdphi + imag_orig[3] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * (a * a + r * r) * (a * a + r * r))));
  imag_conv[3] = factor * (imag_orig[3] * cosmdphi - real_orig[3] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * (a * a + r * r) * (a * a + r * r))));
  real_conv[4] =
      factor * (real_orig[4] * cosmdphi + imag_orig[4] * sinmdphi) *
      (-0.5 *
       ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
        (a * a + (-2 + r) * r) * (a * a + 2 * r * r + a * a * cos_2th)) /
       (r * r * r * (a * a + r * r) * (a * a + r * r)));
  imag_conv[4] =
      factor * (imag_orig[4] * cosmdphi - real_orig[4] * sinmdphi) *
      (-0.5 *
       ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
        (a * a + (-2 + r) * r) * (a * a + 2 * r * r + a * a * cos_2th)) /
       (r * r * r * (a * a + r * r) * (a * a + r * r)));
  real_conv[5] = factor * (real_orig[5] * cosmdphi + imag_orig[5] * sinmdphi) *
                 (-0.5 *
                  ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                   (a * a + 2 * r * r + a * a * cos_2th)) /
                  (r * r * (a * a + r * r) * (a * a + r * r)));
  imag_conv[5] = factor * (imag_orig[5] * cosmdphi - real_orig[5] * sinmdphi) *
                 (-0.5 *
                  ((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                   (a * a + 2 * r * r + a * a * cos_2th)) /
                  (r * r * (a * a + r * r) * (a * a + r * r)));
  real_conv[6] = factor * (real_orig[6] * cosmdphi + imag_orig[6] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                     (r * r + a * a * cos_sq_th)) /
                    (sinth * r * r * (a * a + r * r) * (a * a + r * r))));
  imag_conv[6] = factor * (imag_orig[6] * cosmdphi - real_orig[6] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (a * a + (-2 + r) * r) *
                     (r * r + a * a * cos_sq_th)) /
                    (sinth * r * r * (a * a + r * r) * (a * a + r * r))));
  real_conv[7] = factor * (real_orig[7] * cosmdphi + imag_orig[7] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (r * (a * a + r * r) * (a * a + r * r))));
  imag_conv[7] = factor * (imag_orig[7] * cosmdphi - real_orig[7] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (r * (a * a + r * r) * (a * a + r * r))));
  real_conv[8] = factor * (real_orig[8] * cosmdphi + imag_orig[8] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * r * (a * a + r * r) * (a * a + r * r))));
  imag_conv[8] = factor * (imag_orig[8] * cosmdphi - real_orig[8] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * r * (a * a + r * r) * (a * a + r * r))));
  real_conv[9] = factor * (real_orig[9] * cosmdphi + imag_orig[9] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * sinth * r * (a * a + r * r) * (a * a + r * r))));
  imag_conv[9] = factor * (imag_orig[9] * cosmdphi - real_orig[9] * sinmdphi) *
                 (-(((a * a + (-2 + r) * r) * (r * r + a * a * cos_sq_th)) /
                    (sinth * sinth * r * (a * a + r * r) * (a * a + r * r))));
}

void convert_effsource_dpsidtheta(const int m, const double a, const double r,
                                  const double th,
                                  std::array<double, 10>& real_orig,
                                  std::array<double, 10>& imag_orig,
                                  std::array<double, 10>& real_orig_dth,
                                  std::array<double, 10>& imag_orig_dth,
                                  std::array<double, 10>& real_conv_dth,
                                  std::array<double, 10>& imag_conv_dth) {
  const double factor = r / (2 * M_PI);
  const double rplus = 1 + sqrt(1 - a * a);
  const double rminus = 1 - sqrt(1 - a * a);
  const double mdphi =
      m * a / (rplus - rminus) * log((r - rplus) / (r - rminus));
  const double cosmdphi = cos(mdphi);
  const double sinmdphi = sin(mdphi);
  const double sinth = sin(th);
  const double costh = cos(th);
  real_conv_dth[0] =
      factor * (real_orig_dth[0] * cosmdphi + imag_orig_dth[0] * sinmdphi);
  imag_conv_dth[0] =
      factor * (imag_orig_dth[0] * cosmdphi - real_orig_dth[0] * sinmdphi);
  real_conv_dth[1] =
      factor * (real_orig_dth[1] * cosmdphi + imag_orig_dth[1] * sinmdphi) *
      (r * r - 2 * r + a * a) / (r * r);
  imag_conv_dth[1] =
      factor * (imag_orig_dth[1] * cosmdphi - real_orig_dth[1] * sinmdphi) *
      (r * r - 2 * r + a * a) / (r * r);
  real_conv_dth[2] =
      factor * (real_orig_dth[2] * cosmdphi + imag_orig_dth[2] * sinmdphi) / r;
  imag_conv_dth[2] =
      factor * (imag_orig_dth[2] * cosmdphi - real_orig_dth[2] * sinmdphi) / r;
  real_conv_dth[3] =
      factor * (real_orig_dth[3] * cosmdphi + imag_orig_dth[3] * sinmdphi) /
          (r * sinth) -
      factor * (real_orig[3] * cosmdphi + imag_orig[3] * sinmdphi) * costh /
          (r * sinth * sinth);
  imag_conv_dth[3] =
      factor * (imag_orig_dth[3] * cosmdphi - real_orig_dth[3] * sinmdphi) /
          (r * sinth) -
      factor * (imag_orig[3] * cosmdphi - real_orig[3] * sinmdphi) * costh /
          (r * sinth * sinth);
  real_conv_dth[4] =
      factor * (real_orig_dth[4] * cosmdphi + imag_orig_dth[4] * sinmdphi) *
      (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) / (r * r * r * r);
  imag_conv_dth[4] =
      factor * (imag_orig_dth[4] * cosmdphi - real_orig_dth[4] * sinmdphi) *
      (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) / (r * r * r * r);
  real_conv_dth[5] =
      factor * (real_orig_dth[5] * cosmdphi + imag_orig_dth[5] * sinmdphi) *
      (r * r - 2 * r + a * a) / (r * r * r);
  imag_conv_dth[5] =
      factor * (imag_orig_dth[5] * cosmdphi - real_orig_dth[5] * sinmdphi) *
      (r * r - 2 * r + a * a) / (r * r * r);
  real_conv_dth[6] =
      factor * (real_orig_dth[6] * cosmdphi + imag_orig_dth[6] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r * sinth) -
      factor * (real_orig[6] * cosmdphi + imag_orig[6] * sinmdphi) *
          (r * r - 2 * r + a * a) * costh / (r * r * r * sinth * sinth);
  imag_conv_dth[6] =
      factor * (imag_orig_dth[6] * cosmdphi - real_orig_dth[6] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r * sinth) -
      factor * (imag_orig[6] * cosmdphi - real_orig[6] * sinmdphi) *
          (r * r - 2 * r + a * a) * costh / (r * r * r * sinth * sinth);
  real_conv_dth[7] =
      factor * (real_orig_dth[7] * cosmdphi + imag_orig_dth[7] * sinmdphi) /
      (r * r);
  imag_conv_dth[7] =
      factor * (imag_orig_dth[7] * cosmdphi - real_orig_dth[7] * sinmdphi) /
      (r * r);
  real_conv_dth[8] =
      factor * (real_orig_dth[8] * cosmdphi + imag_orig_dth[8] * sinmdphi) /
          (r * r * sinth) -
      factor * (real_orig[8] * cosmdphi + imag_orig[8] * sinmdphi) * costh /
          (r * r * sinth * sinth);
  imag_conv_dth[8] =
      factor * (imag_orig_dth[8] * cosmdphi - real_orig_dth[8] * sinmdphi) /
          (r * r * sinth) -
      factor * (imag_orig[8] * cosmdphi - real_orig[8] * sinmdphi) * costh /
          (r * r * sinth * sinth);
  real_conv_dth[9] =
      factor * (real_orig_dth[9] * cosmdphi + imag_orig_dth[9] * sinmdphi) /
          (r * r * sinth * sinth) -
      factor * (real_orig[9] * cosmdphi + imag_orig[9] * sinmdphi) * 2 * costh /
          (r * r * sinth * sinth * sinth);
  imag_conv_dth[9] =
      factor * (imag_orig_dth[9] * cosmdphi - real_orig_dth[9] * sinmdphi) /
          (r * r * sinth * sinth) -
      factor * (imag_orig[9] * cosmdphi - real_orig[9] * sinmdphi) * 2 * costh /
          (r * r * sinth * sinth * sinth);
}

void convert_effsource_dpsidrstar(const int m, const double a, const double r,
                                  const double th,
                                  std::array<double, 10>& real_orig,
                                  std::array<double, 10>& imag_orig,
                                  std::array<double, 10>& real_orig_dr,
                                  std::array<double, 10>& imag_orig_dr,
                                  std::array<double, 10>& real_conv_drs,
                                  std::array<double, 10>& imag_conv_drs) {
  const double factor =
      (r * r - 2 * r + a * a) * r / ((r * r + a * a) * 2 * M_PI);
  const double rplus = 1 + sqrt(1 - a * a);
  const double rminus = 1 - sqrt(1 - a * a);
  const double mdphi =
      m * a / (rplus - rminus) * log((r - rplus) / (r - rminus));
  const double mdphi_dr = m * a / (r * r - 2 * r + a * a);
  const double cosmdphi = cos(mdphi);
  const double sinmdphi = sin(mdphi);
  const double sinth = sin(th);
  real_conv_drs[0] =
      factor * (real_orig_dr[0] * cosmdphi + imag_orig_dr[0] * sinmdphi) +
      factor * (real_orig[0] * cosmdphi + imag_orig[0] * sinmdphi) / r +
      factor * (imag_orig[0] * cosmdphi - real_orig[0] * sinmdphi) * (mdphi_dr);
  imag_conv_drs[0] =
      factor * (imag_orig_dr[0] * cosmdphi - real_orig_dr[0] * sinmdphi) +
      factor * (imag_orig[0] * cosmdphi - real_orig[0] * sinmdphi) / r +
      factor * (-real_orig[0] * cosmdphi - imag_orig[0] * sinmdphi) *
          (mdphi_dr);
  real_conv_drs[1] =
      factor * (real_orig_dr[1] * cosmdphi + imag_orig_dr[1] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r) +
      factor * (real_orig[1] * cosmdphi + imag_orig[1] * sinmdphi) *
          (r * r - a * a) / (r * r * r) +
      factor * (imag_orig[1] * cosmdphi - real_orig[1] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (r * r));
  imag_conv_drs[1] =
      factor * (imag_orig_dr[1] * cosmdphi - real_orig_dr[1] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r) +
      factor * (imag_orig[1] * cosmdphi - real_orig[1] * sinmdphi) *
          (r * r - a * a) / (r * r * r) +
      factor * (-real_orig[1] * cosmdphi - imag_orig[1] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (r * r));
  real_conv_drs[2] =
      factor * (real_orig_dr[2] * cosmdphi + imag_orig_dr[2] * sinmdphi) / r +
      factor * (imag_orig[2] * cosmdphi - real_orig[2] * sinmdphi) *
          (mdphi_dr / r);
  imag_conv_drs[2] =
      factor * (imag_orig_dr[2] * cosmdphi - real_orig_dr[2] * sinmdphi) / r +
      factor * (-real_orig[2] * cosmdphi - imag_orig[2] * sinmdphi) *
          (mdphi_dr / r);
  real_conv_drs[3] =
      factor * (real_orig_dr[3] * cosmdphi + imag_orig_dr[3] * sinmdphi) /
          (r * sinth) +
      factor * (imag_orig[3] * cosmdphi - real_orig[3] * sinmdphi) *
          (mdphi_dr / (r * sinth));
  imag_conv_drs[3] =
      factor * (imag_orig_dr[3] * cosmdphi - real_orig_dr[3] * sinmdphi) /
          (r * sinth) +
      factor * (-real_orig[3] * cosmdphi - imag_orig[3] * sinmdphi) *
          (mdphi_dr / (r * sinth));
  real_conv_drs[4] =
      factor * (real_orig_dr[4] * cosmdphi + imag_orig_dr[4] * sinmdphi) *
          (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) / (r * r * r * r) +
      factor * (real_orig[4] * cosmdphi + imag_orig[4] * sinmdphi) *
          (r * r - 2 * r + a * a) * (r * (2 + r) - 3 * a * a) /
          (r * r * r * r * r) +
      factor * (imag_orig[4] * cosmdphi - real_orig[4] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) /
           (r * r * r * r));
  imag_conv_drs[4] =
      factor * (imag_orig_dr[4] * cosmdphi - real_orig_dr[4] * sinmdphi) *
          (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) / (r * r * r * r) +
      factor * (imag_orig[4] * cosmdphi - real_orig[4] * sinmdphi) *
          (r * r - 2 * r + a * a) * (r * (2 + r) - 3 * a * a) /
          (r * r * r * r * r) +
      factor * (-real_orig[4] * cosmdphi - imag_orig[4] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) * (r * r - 2 * r + a * a) /
           (r * r * r * r));
  real_conv_drs[5] =
      factor * (real_orig_dr[5] * cosmdphi + imag_orig_dr[5] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r) +
      factor * (real_orig[5] * cosmdphi + imag_orig[5] * sinmdphi) * 2 *
          (r - a * a) / (r * r * r * r) +
      factor * (imag_orig[5] * cosmdphi - real_orig[5] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (r * r * r));
  imag_conv_drs[5] =
      factor * (imag_orig_dr[5] * cosmdphi - real_orig_dr[5] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r) +
      factor * (imag_orig[5] * cosmdphi - real_orig[5] * sinmdphi) * 2 *
          (r - a * a) / (r * r * r * r) +
      factor * (-real_orig[5] * cosmdphi - imag_orig[5] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (r * r * r));
  real_conv_drs[6] =
      factor * (real_orig_dr[6] * cosmdphi + imag_orig_dr[6] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r * sinth) +
      factor * (real_orig[6] * cosmdphi + imag_orig[6] * sinmdphi) * 2 *
          (r - a * a) / (sinth * r * r * r * r) +
      factor * (imag_orig[6] * cosmdphi - real_orig[6] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (sinth * r * r * r));
  imag_conv_drs[6] =
      factor * (imag_orig_dr[6] * cosmdphi - real_orig_dr[6] * sinmdphi) *
          (r * r - 2 * r + a * a) / (r * r * r * sinth) +
      factor * (imag_orig[6] * cosmdphi - real_orig[6] * sinmdphi) * 2 *
          (r - a * a) / (sinth * r * r * r * r) +
      factor * (-real_orig[6] * cosmdphi - imag_orig[6] * sinmdphi) *
          (mdphi_dr * (r * r - 2 * r + a * a) / (sinth * r * r * r));
  real_conv_drs[7] =
      factor * (real_orig_dr[7] * cosmdphi + imag_orig_dr[7] * sinmdphi) /
          (r * r) +
      factor * (real_orig[7] * cosmdphi + imag_orig[7] * sinmdphi) /
          (-r * r * r) +
      factor * (imag_orig[7] * cosmdphi - real_orig[7] * sinmdphi) *
          (mdphi_dr / (r * r));
  imag_conv_drs[7] =
      factor * (imag_orig_dr[7] * cosmdphi - real_orig_dr[7] * sinmdphi) /
          (r * r) +
      factor * (imag_orig[7] * cosmdphi - real_orig[7] * sinmdphi) /
          (-r * r * r) +
      factor * (-real_orig[7] * cosmdphi - imag_orig[7] * sinmdphi) *
          (mdphi_dr / (r * r));
  real_conv_drs[8] =
      factor * (real_orig_dr[8] * cosmdphi + imag_orig_dr[8] * sinmdphi) /
          (r * r * sinth) +
      factor * (real_orig[8] * cosmdphi + imag_orig[8] * sinmdphi) /
          (-r * r * r * sinth) +
      factor * (imag_orig[8] * cosmdphi - real_orig[8] * sinmdphi) *
          (mdphi_dr / (r * r * sinth));
  imag_conv_drs[8] =
      factor * (imag_orig_dr[8] * cosmdphi - real_orig_dr[8] * sinmdphi) /
          (r * r * sinth) +
      factor * (imag_orig[8] * cosmdphi - real_orig[8] * sinmdphi) /
          (-r * r * r * sinth) +
      factor * (-real_orig[8] * cosmdphi - imag_orig[8] * sinmdphi) *
          (mdphi_dr / (r * r * sinth));
  real_conv_drs[9] =
      factor * (real_orig_dr[9] * cosmdphi + imag_orig_dr[9] * sinmdphi) /
          (r * r * sinth * sinth) +
      factor * (real_orig[9] * cosmdphi + imag_orig[9] * sinmdphi) /
          (-r * r * r * sinth * sinth) +
      factor * (imag_orig[9] * cosmdphi - real_orig[9] * sinmdphi) *
          (mdphi_dr / (r * r * sinth * sinth));
  imag_conv_drs[9] =
      factor * (imag_orig_dr[9] * cosmdphi - real_orig_dr[9] * sinmdphi) /
          (r * r * sinth * sinth) +
      factor * (imag_orig[9] * cosmdphi - real_orig[9] * sinmdphi) /
          (-r * r * r * sinth * sinth) +
      factor * (-real_orig[9] * cosmdphi - imag_orig[9] * sinmdphi) *
          (mdphi_dr / (r * r * sinth * sinth));
}

void convert_effsource_psi_vr(int m, double a, double r, double z,
                           std::array<double, 10>& real_orig,
                           std::array<double, 10>& imag_orig,
                           std::array<double, 10>& real_conv,
                           std::array<double, 10>& imag_conv){
double factor = 1.0/(2*M_PI);
double sinth = sqrt(1-z*z);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
real_conv[0] = factor*((real_orig[0])*cosmdphi+(imag_orig[0])*sinmdphi)*r;
imag_conv[0] = factor*((imag_orig[0])*cosmdphi-(real_orig[0])*sinmdphi)*r;
real_conv[1] = factor*((real_orig[1]-(real_orig[3]*a)/(a*a-2*r+r*r)+(real_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[1]-(imag_orig[3]*a)/(a*a-2*r+r*r)+(imag_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
imag_conv[1] = factor*((imag_orig[1]-(imag_orig[3]*a)/(a*a-2*r+r*r)+(imag_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[1]-(real_orig[3]*a)/(a*a-2*r+r*r)+(real_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
real_conv[2] = factor*((real_orig[2])*cosmdphi+(imag_orig[2])*sinmdphi);
imag_conv[2] = factor*((imag_orig[2])*cosmdphi-(real_orig[2])*sinmdphi);
real_conv[3] = factor*((real_orig[3])*cosmdphi+(imag_orig[3])*sinmdphi)/sinth;
imag_conv[3] = factor*((imag_orig[3])*cosmdphi-(real_orig[3])*sinmdphi)/sinth;
real_conv[4] = factor*((real_orig[4]+(real_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig[6]*a)/(a*a-2*r+r*r)+(2*real_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[4]+(imag_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig[6]*a)/(a*a-2*r+r*r)+(2*imag_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
imag_conv[4] = factor*((imag_orig[4]+(imag_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig[6]*a)/(a*a-2*r+r*r)+(2*imag_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[4]+(real_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig[6]*a)/(a*a-2*r+r*r)+(2*real_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
real_conv[5] = factor*((real_orig[5]-(real_orig[8]*a)/(a*a-2*r+r*r)+(real_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[5]-(imag_orig[8]*a)/(a*a-2*r+r*r)+(imag_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi);
imag_conv[5] = factor*((imag_orig[5]-(imag_orig[8]*a)/(a*a-2*r+r*r)+(imag_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[5]-(real_orig[8]*a)/(a*a-2*r+r*r)+(real_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi);
real_conv[6] = factor*((real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/sinth;
imag_conv[6] = factor*((imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/sinth;
real_conv[7] = factor*(real_orig[7]*cosmdphi+imag_orig[7]*sinmdphi)/r;
imag_conv[7] = factor*(imag_orig[7]*cosmdphi-real_orig[7]*sinmdphi)/r;
real_conv[8] = factor*(real_orig[8]*cosmdphi+imag_orig[8]*sinmdphi)/(r*sinth);
imag_conv[8] = factor*(imag_orig[8]*cosmdphi-real_orig[8]*sinmdphi)/(r*sinth);
real_conv[9] = factor*(real_orig[9]*cosmdphi+imag_orig[9]*sinmdphi)/(r*sinth*sinth);
imag_conv[9] = factor*(imag_orig[9]*cosmdphi-real_orig[9]*sinmdphi)/(r*sinth*sinth);
}

void convert_effsource_Seff_vr(int m, double a, double r, double z,
                            std::array<double, 10>& real_orig,
                            std::array<double, 10>& imag_orig,
                            std::array<double, 10>& real_conv,
                            std::array<double, 10>& imag_conv){
double factor = 1.0/(2*M_PI);
double sinth = sqrt(1-z*z);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
real_conv[0] = factor*(real_orig[0]*cosmdphi+imag_orig[0]*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
imag_conv[0] = factor*(imag_orig[0]*cosmdphi-real_orig[0]*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
real_conv[1] = factor*((real_orig[1]-(real_orig[3]*a)/(a*a-2*r+r*r)+(real_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[1]-(imag_orig[3]*a)/(a*a-2*r+r*r)+(imag_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
imag_conv[1] = factor*((imag_orig[1]-(imag_orig[3]*a)/(a*a-2*r+r*r)+(imag_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[1]-(real_orig[3]*a)/(a*a-2*r+r*r)+(real_orig[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
real_conv[2] = factor*(real_orig[2]*cosmdphi+imag_orig[2]*sinmdphi)*(r*r + a*a*z*z)/(a*a + r*r);
imag_conv[2] = factor*(imag_orig[2]*cosmdphi-real_orig[2]*sinmdphi)*(r*r + a*a*z*z)/(a*a + r*r);
real_conv[3] = factor*(real_orig[3]*cosmdphi+imag_orig[3]*sinmdphi)*((r*r + a*a*z*z))/(sinth*(a*a + r*r));
imag_conv[3] = factor*(imag_orig[3]*cosmdphi-real_orig[3]*sinmdphi)*((r*r + a*a*z*z))/(sinth*(a*a + r*r));
real_conv[4] = factor*((real_orig[4]+(real_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig[6]*a)/(a*a-2*r+r*r)+(2*real_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[4]+(imag_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig[6]*a)/(a*a-2*r+r*r)+(2*imag_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
imag_conv[4] = factor*((imag_orig[4]+(imag_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig[6]*a)/(a*a-2*r+r*r)+(2*imag_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[4]+(real_orig[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig[6]*a)/(a*a-2*r+r*r)+(2*real_orig[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*(r*r + a*a*z*z))/(a*a + r*r);
real_conv[5] = factor*((real_orig[5]-(real_orig[8]*a)/(a*a-2*r+r*r)+(real_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[5]-(imag_orig[8]*a)/(a*a-2*r+r*r)+(imag_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*r + a*a*z*z)/(a*a + r*r);
imag_conv[5] = factor*((imag_orig[5]-(imag_orig[8]*a)/(a*a-2*r+r*r)+(imag_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[5]-(real_orig[8]*a)/(a*a-2*r+r*r)+(real_orig[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*(r*r + a*a*z*z)/(a*a + r*r);
real_conv[6] = factor*((real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*((r*r + a*a*z*z))/(sinth*(a*a + r*r));
imag_conv[6] = factor*((imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*((r*r + a*a*z*z))/(sinth*(a*a + r*r));
real_conv[7] = factor*(real_orig[7]*cosmdphi+imag_orig[7]*sinmdphi)*(r*r + a*a*z*z)/(r*(a*a + r*r));
imag_conv[7] = factor*(imag_orig[7]*cosmdphi-real_orig[7]*sinmdphi)*(r*r + a*a*z*z)/(r*(a*a + r*r));
real_conv[8] = factor*(real_orig[8]*cosmdphi+imag_orig[8]*sinmdphi)*((r*r + a*a*z*z))/(r*sinth*(a*a + r*r));
imag_conv[8] = factor*(imag_orig[8]*cosmdphi-real_orig[8]*sinmdphi)*((r*r + a*a*z*z))/(r*sinth*(a*a + r*r));
real_conv[9] = factor*(real_orig[9]*cosmdphi+imag_orig[9]*sinmdphi)*((r*r + a*a*z*z))/(r*sinth*sinth*(a*a + r*r));
imag_conv[9] = factor*(imag_orig[9]*cosmdphi-real_orig[9]*sinmdphi)*((r*r + a*a*z*z))/(r*sinth*sinth*(a*a + r*r));
}

void convert_effsource_dpsidz_vr(int m, double a, double r, double z,
                                  std::array<double, 10>& real_orig,
                                  std::array<double, 10>& imag_orig,
                                  std::array<double, 10>& real_orig_dth,
                                  std::array<double, 10>& imag_orig_dth,
                                  std::array<double, 10>& real_conv_dz,
                                  std::array<double, 10>& imag_conv_dz){
double sinth = sqrt(1-z*z);
double dthdz = -1.0/sinth;
double factor = dthdz/(2*M_PI);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
real_conv_dz[0] = factor*((real_orig_dth[0])*cosmdphi+(imag_orig_dth[0])*sinmdphi)*r;
imag_conv_dz[0] = factor*((imag_orig_dth[0])*cosmdphi-(real_orig_dth[0])*sinmdphi)*r;
real_conv_dz[1] = factor*((real_orig_dth[1]-(real_orig_dth[3]*a)/(a*a-2*r+r*r)+(real_orig_dth[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig_dth[1]-(imag_orig_dth[3]*a)/(a*a-2*r+r*r)+(imag_orig_dth[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
imag_conv_dz[1] = factor*((imag_orig_dth[1]-(imag_orig_dth[3]*a)/(a*a-2*r+r*r)+(imag_orig_dth[0]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig_dth[1]-(real_orig_dth[3]*a)/(a*a-2*r+r*r)+(real_orig_dth[0]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
real_conv_dz[2] = factor*((real_orig_dth[2])*cosmdphi+(imag_orig_dth[2])*sinmdphi);
imag_conv_dz[2] = factor*((imag_orig_dth[2])*cosmdphi-(real_orig_dth[2])*sinmdphi);
real_conv_dz[3] = factor*(((real_orig_dth[3])*cosmdphi+(imag_orig_dth[3])*sinmdphi)/sinth - z*((real_orig[3])*cosmdphi+(imag_orig[3])*sinmdphi)/(sinth*sinth));
imag_conv_dz[3] = factor*(((imag_orig_dth[3])*cosmdphi-(real_orig_dth[3])*sinmdphi)/sinth - z*((imag_orig[3])*cosmdphi-(real_orig[3])*sinmdphi)/(sinth*sinth));
real_conv_dz[4] = factor*((real_orig_dth[4]+(real_orig_dth[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig_dth[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig_dth[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig_dth[6]*a)/(a*a-2*r+r*r)+(2*real_orig_dth[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig_dth[4]+(imag_orig_dth[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig_dth[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig_dth[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig_dth[6]*a)/(a*a-2*r+r*r)+(2*imag_orig_dth[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
imag_conv_dz[4] = factor*((imag_orig_dth[4]+(imag_orig_dth[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*imag_orig_dth[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(imag_orig_dth[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*imag_orig_dth[6]*a)/(a*a-2*r+r*r)+(2*imag_orig_dth[1]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig_dth[4]+(real_orig_dth[9]*a*a)/pow(a*a-2*r+r*r,2)-(2*real_orig_dth[3]*a*(-a*a-r*r))/pow(a*a-2*r+r*r,2)+(real_orig_dth[0]*pow(-a*a-r*r,2))/pow(a*a-2*r+r*r,2)-(2*real_orig_dth[6]*a)/(a*a-2*r+r*r)+(2*real_orig_dth[1]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)*r;
real_conv_dz[5] = factor*((real_orig_dth[5]-(real_orig_dth[8]*a)/(a*a-2*r+r*r)+(real_orig_dth[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig_dth[5]-(imag_orig_dth[8]*a)/(a*a-2*r+r*r)+(imag_orig_dth[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi);
imag_conv_dz[5] = factor*((imag_orig_dth[5]-(imag_orig_dth[8]*a)/(a*a-2*r+r*r)+(imag_orig_dth[2]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig_dth[5]-(real_orig_dth[8]*a)/(a*a-2*r+r*r)+(real_orig_dth[2]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi);
real_conv_dz[6] = factor*(((real_orig_dth[6]-(real_orig_dth[9]*a)/(a*a-2*r+r*r)+(real_orig_dth[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig_dth[6]-(imag_orig_dth[9]*a)/(a*a-2*r+r*r)+(imag_orig_dth[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/sinth - z*((real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi+(imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/(sinth*sinth) );
imag_conv_dz[6] = factor*(((imag_orig_dth[6]-(imag_orig_dth[9]*a)/(a*a-2*r+r*r)+(imag_orig_dth[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig_dth[6]-(real_orig_dth[9]*a)/(a*a-2*r+r*r)+(real_orig_dth[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/sinth - z*((imag_orig[6]-(imag_orig[9]*a)/(a*a-2*r+r*r)+(imag_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*cosmdphi-(real_orig[6]-(real_orig[9]*a)/(a*a-2*r+r*r)+(real_orig[3]*(-a*a-r*r))/(a*a-2*r+r*r))*sinmdphi)/(sinth*sinth) );
real_conv_dz[7] = factor*(real_orig_dth[7]*cosmdphi+imag_orig_dth[7]*sinmdphi)/r;
imag_conv_dz[7] = factor*(imag_orig_dth[7]*cosmdphi-real_orig_dth[7]*sinmdphi)/r;
real_conv_dz[8] = factor*((real_orig_dth[8]*cosmdphi+imag_orig_dth[8]*sinmdphi)/(r*sinth) - z*(real_orig[8]*cosmdphi+imag_orig[8]*sinmdphi)/(r*sinth*sinth) );
imag_conv_dz[8] = factor*((imag_orig_dth[8]*cosmdphi-real_orig_dth[8]*sinmdphi)/(r*sinth) - z*(imag_orig[8]*cosmdphi-real_orig[8]*sinmdphi)/(r*sinth*sinth) );
real_conv_dz[9] = factor*((real_orig_dth[9]*cosmdphi+imag_orig_dth[9]*sinmdphi)/(r*sinth*sinth) - 2*z*(real_orig[9]*cosmdphi+imag_orig[9]*sinmdphi)/(r*sinth*sinth*sinth) );
imag_conv_dz[9] = factor*((imag_orig_dth[9]*cosmdphi-real_orig_dth[9]*sinmdphi)/(r*sinth*sinth) - 2*z*(imag_orig[9]*cosmdphi-real_orig[9]*sinmdphi)/(r*sinth*sinth*sinth) );
}

void convert_effsource_dpsidr_vr(int m, double a, double r, double z,
                                  std::array<double, 10>& real_orig,
                                  std::array<double, 10>& imag_orig,
                                  std::array<double, 10>& real_orig_dr,
                                  std::array<double, 10>& imag_orig_dr,
                                  std::array<double, 10>& real_conv_dr,
                                  std::array<double, 10>& imag_conv_dr){
double factor = 1.0/(2*M_PI);
double sinth = sqrt(1-z*z);
double rplus = 1+sqrt(1-a*a);
double rminus = 1-sqrt(1-a*a);
double mdphi = m*a/(rplus-rminus)*log((r-rplus)/(r-rminus));
double mdphi_dr = m*a/(r*r-2*r+a*a);
double cosmdphi = cos(mdphi);
double sinmdphi = sin(mdphi);
real_conv_dr[0]=factor*(cosmdphi*(real_orig[0]+real_orig_dr[0]*r)+(imag_orig[0]+imag_orig_dr[0]*r)*sinmdphi+mdphi_dr*r*(imag_orig[0]*cosmdphi-real_orig[0]*sinmdphi));
imag_conv_dr[0]=-(factor*(-(cosmdphi*(imag_orig[0]+imag_orig_dr[0]*r))+(real_orig[0]+real_orig_dr[0]*r)*sinmdphi+mdphi_dr*r*(real_orig[0]*cosmdphi+imag_orig[0]*sinmdphi)));
real_conv_dr[1]=-((factor*(cosmdphi*(a*a+(-2+r)*r)*(real_orig[3]*a-real_orig[1]*(a*a+(-2+r)*r)+real_orig[0]*(a*a+r*r))+(a*a+(-2+r)*r)*(imag_orig[3]*a-imag_orig[1]*(a*a+(-2+r)*r)+imag_orig[0]*(a*a+r*r))*sinmdphi+r*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(imag_orig[3]*a-imag_orig[1]*(a*a+(-2+r)*r)+imag_orig[0]*(a*a+r*r))+cosmdphi*(-2*real_orig[3]*a*(-1+r)+real_orig_dr[3]*a*(a*a+(-2+r)*r)+2*real_orig[0]*r*(a*a+(-2+r)*r)-real_orig_dr[1]*pow(a*a+(-2+r)*r,2)-2*real_orig[0]*(-1+r)*(a*a+r*r)+real_orig_dr[0]*(a*a+(-2+r)*r)*(a*a+r*r))-mdphi_dr*(a*a+(-2+r)*r)*(real_orig[3]*a-real_orig[1]*(a*a+(-2+r)*r)+real_orig[0]*(a*a+r*r))*sinmdphi+(-2*imag_orig[3]*a*(-1+r)+imag_orig_dr[3]*a*(a*a+(-2+r)*r)+2*imag_orig[0]*r*(a*a+(-2+r)*r)-imag_orig_dr[1]*pow(a*a+(-2+r)*r,2)-2*imag_orig[0]*(-1+r)*(a*a+r*r)+imag_orig_dr[0]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi)))/pow(a*a+(-2+r)*r,2));
imag_conv_dr[1]=(factor*(-(cosmdphi*(a*a+(-2+r)*r)*(imag_orig[3]*a-imag_orig[1]*(a*a+(-2+r)*r)+imag_orig[0]*(a*a+r*r)))+(a*a+(-2+r)*r)*(real_orig[3]*a-real_orig[1]*(a*a+(-2+r)*r)+real_orig[0]*(a*a+r*r))*sinmdphi+r*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(real_orig[3]*a-real_orig[1]*(a*a+(-2+r)*r)+real_orig[0]*(a*a+r*r))-cosmdphi*(-2*imag_orig[3]*a*(-1+r)+imag_orig_dr[3]*a*(a*a+(-2+r)*r)+2*imag_orig[0]*r*(a*a+(-2+r)*r)-imag_orig_dr[1]*pow(a*a+(-2+r)*r,2)-2*imag_orig[0]*(-1+r)*(a*a+r*r)+imag_orig_dr[0]*(a*a+(-2+r)*r)*(a*a+r*r))+mdphi_dr*(a*a+(-2+r)*r)*(imag_orig[3]*a-imag_orig[1]*(a*a+(-2+r)*r)+imag_orig[0]*(a*a+r*r))*sinmdphi+(-2*real_orig[3]*a*(-1+r)+real_orig_dr[3]*a*(a*a+(-2+r)*r)+2*real_orig[0]*r*(a*a+(-2+r)*r)-real_orig_dr[1]*pow(a*a+(-2+r)*r,2)-2*real_orig[0]*(-1+r)*(a*a+r*r)+real_orig_dr[0]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi)))/pow(a*a+(-2+r)*r,2);
real_conv_dr[2]=factor*(real_orig_dr[2]*cosmdphi+imag_orig_dr[2]*sinmdphi+mdphi_dr*(imag_orig[2]*cosmdphi-real_orig[2]*sinmdphi));
imag_conv_dr[2]=-(factor*(-(imag_orig_dr[2]*cosmdphi)+real_orig_dr[2]*sinmdphi+mdphi_dr*(real_orig[2]*cosmdphi+imag_orig[2]*sinmdphi)));
real_conv_dr[3]=(factor*(real_orig_dr[3]*cosmdphi+imag_orig_dr[3]*sinmdphi+mdphi_dr*(imag_orig[3]*cosmdphi-real_orig[3]*sinmdphi)))/sinth;
imag_conv_dr[3]=-((factor*(-(imag_orig_dr[3]*cosmdphi)+real_orig_dr[3]*sinmdphi+mdphi_dr*(real_orig[3]*cosmdphi+imag_orig[3]*sinmdphi)))/sinth);
real_conv_dr[4]=(factor*(cosmdphi*(a*a+(-2+r)*r)*(real_orig[9]*a*a-2*real_orig[6]*a*(a*a+(-2+r)*r)+real_orig[4]*pow(a*a+(-2+r)*r,2)+2*real_orig[3]*a*(a*a+r*r)-2*real_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+real_orig[0]*pow(a*a+r*r,2))+(a*a+(-2+r)*r)*(imag_orig[9]*a*a-2*imag_orig[6]*a*(a*a+(-2+r)*r)+imag_orig[4]*pow(a*a+(-2+r)*r,2)+2*imag_orig[3]*a*(a*a+r*r)-2*imag_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+imag_orig[0]*pow(a*a+r*r,2))*sinmdphi+r*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(imag_orig[9]*a*a-2*imag_orig[6]*a*(a*a+(-2+r)*r)+imag_orig[4]*pow(a*a+(-2+r)*r,2)+2*imag_orig[3]*a*(a*a+r*r)-2*imag_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+imag_orig[0]*pow(a*a+r*r,2))+cosmdphi*(-4*real_orig[9]*a*a*(-1+r)+real_orig_dr[9]*a*a*(a*a+(-2+r)*r)+4*real_orig[6]*a*(-1+r)*(a*a+(-2+r)*r)+4*real_orig[3]*a*r*(a*a+(-2+r)*r)-2*real_orig_dr[6]*a*pow(a*a+(-2+r)*r,2)-4*real_orig[1]*r*pow(a*a+(-2+r)*r,2)+real_orig_dr[4]*pow(a*a+(-2+r)*r,3)-8*real_orig[3]*a*(-1+r)*(a*a+r*r)+2*real_orig_dr[3]*a*(a*a+(-2+r)*r)*(a*a+r*r)+4*real_orig[1]*(-1+r)*(a*a+(-2+r)*r)*(a*a+r*r)+4*real_orig[0]*r*(a*a+(-2+r)*r)*(a*a+r*r)-2*real_orig_dr[1]*pow(a*a+(-2+r)*r,2)*(a*a+r*r)-4*real_orig[0]*(-1+r)*pow(a*a+r*r,2)+real_orig_dr[0]*(a*a+(-2+r)*r)*pow(a*a+r*r,2))-mdphi_dr*(a*a+(-2+r)*r)*(real_orig[9]*a*a-2*real_orig[6]*a*(a*a+(-2+r)*r)+real_orig[4]*pow(a*a+(-2+r)*r,2)+2*real_orig[3]*a*(a*a+r*r)-2*real_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+real_orig[0]*pow(a*a+r*r,2))*sinmdphi+(-4*imag_orig[9]*a*a*(-1+r)+imag_orig_dr[9]*a*a*(a*a+(-2+r)*r)+4*imag_orig[6]*a*(-1+r)*(a*a+(-2+r)*r)+4*imag_orig[3]*a*r*(a*a+(-2+r)*r)-2*imag_orig_dr[6]*a*pow(a*a+(-2+r)*r,2)-4*imag_orig[1]*r*pow(a*a+(-2+r)*r,2)+imag_orig_dr[4]*pow(a*a+(-2+r)*r,3)-8*imag_orig[3]*a*(-1+r)*(a*a+r*r)+2*imag_orig_dr[3]*a*(a*a+(-2+r)*r)*(a*a+r*r)+4*imag_orig[1]*(-1+r)*(a*a+(-2+r)*r)*(a*a+r*r)+4*imag_orig[0]*r*(a*a+(-2+r)*r)*(a*a+r*r)-2*imag_orig_dr[1]*pow(a*a+(-2+r)*r,2)*(a*a+r*r)-4*imag_orig[0]*(-1+r)*pow(a*a+r*r,2)+imag_orig_dr[0]*(a*a+(-2+r)*r)*pow(a*a+r*r,2))*sinmdphi)))/pow(a*a+(-2+r)*r,3);
imag_conv_dr[4]=(factor*(cosmdphi*(a*a+(-2+r)*r)*(imag_orig[9]*a*a-2*imag_orig[6]*a*(a*a+(-2+r)*r)+imag_orig[4]*pow(a*a+(-2+r)*r,2)+2*imag_orig[3]*a*(a*a+r*r)-2*imag_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+imag_orig[0]*pow(a*a+r*r,2))-(a*a+(-2+r)*r)*(real_orig[9]*a*a-2*real_orig[6]*a*(a*a+(-2+r)*r)+real_orig[4]*pow(a*a+(-2+r)*r,2)+2*real_orig[3]*a*(a*a+r*r)-2*real_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+real_orig[0]*pow(a*a+r*r,2))*sinmdphi-r*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(real_orig[9]*a*a-2*real_orig[6]*a*(a*a+(-2+r)*r)+real_orig[4]*pow(a*a+(-2+r)*r,2)+2*real_orig[3]*a*(a*a+r*r)-2*real_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+real_orig[0]*pow(a*a+r*r,2))-cosmdphi*(-4*imag_orig[9]*a*a*(-1+r)+imag_orig_dr[9]*a*a*(a*a+(-2+r)*r)+4*imag_orig[6]*a*(-1+r)*(a*a+(-2+r)*r)+4*imag_orig[3]*a*r*(a*a+(-2+r)*r)-2*imag_orig_dr[6]*a*pow(a*a+(-2+r)*r,2)-4*imag_orig[1]*r*pow(a*a+(-2+r)*r,2)+imag_orig_dr[4]*pow(a*a+(-2+r)*r,3)-8*imag_orig[3]*a*(-1+r)*(a*a+r*r)+2*imag_orig_dr[3]*a*(a*a+(-2+r)*r)*(a*a+r*r)+4*imag_orig[1]*(-1+r)*(a*a+(-2+r)*r)*(a*a+r*r)+4*imag_orig[0]*r*(a*a+(-2+r)*r)*(a*a+r*r)-2*imag_orig_dr[1]*pow(a*a+(-2+r)*r,2)*(a*a+r*r)-4*imag_orig[0]*(-1+r)*pow(a*a+r*r,2)+imag_orig_dr[0]*(a*a+(-2+r)*r)*pow(a*a+r*r,2))+mdphi_dr*(a*a+(-2+r)*r)*(imag_orig[9]*a*a-2*imag_orig[6]*a*(a*a+(-2+r)*r)+imag_orig[4]*pow(a*a+(-2+r)*r,2)+2*imag_orig[3]*a*(a*a+r*r)-2*imag_orig[1]*(a*a+(-2+r)*r)*(a*a+r*r)+imag_orig[0]*pow(a*a+r*r,2))*sinmdphi+(-4*real_orig[9]*a*a*(-1+r)+real_orig_dr[9]*a*a*(a*a+(-2+r)*r)+4*real_orig[6]*a*(-1+r)*(a*a+(-2+r)*r)+4*real_orig[3]*a*r*(a*a+(-2+r)*r)-2*real_orig_dr[6]*a*pow(a*a+(-2+r)*r,2)-4*real_orig[1]*r*pow(a*a+(-2+r)*r,2)+real_orig_dr[4]*pow(a*a+(-2+r)*r,3)-8*real_orig[3]*a*(-1+r)*(a*a+r*r)+2*real_orig_dr[3]*a*(a*a+(-2+r)*r)*(a*a+r*r)+4*real_orig[1]*(-1+r)*(a*a+(-2+r)*r)*(a*a+r*r)+4*real_orig[0]*r*(a*a+(-2+r)*r)*(a*a+r*r)-2*real_orig_dr[1]*pow(a*a+(-2+r)*r,2)*(a*a+r*r)-4*real_orig[0]*(-1+r)*pow(a*a+r*r,2)+real_orig_dr[0]*(a*a+(-2+r)*r)*pow(a*a+r*r,2))*sinmdphi)))/pow(a*a+(-2+r)*r,3);
real_conv_dr[5]=-((factor*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(imag_orig[8]*a-imag_orig[5]*(a*a+(-2+r)*r)+imag_orig[2]*(a*a+r*r))+cosmdphi*(-2*real_orig[8]*a*(-1+r)+real_orig_dr[8]*a*(a*a+(-2+r)*r)+2*real_orig[2]*r*(a*a+(-2+r)*r)-real_orig_dr[5]*pow(a*a+(-2+r)*r,2)-2*real_orig[2]*(-1+r)*(a*a+r*r)+real_orig_dr[2]*(a*a+(-2+r)*r)*(a*a+r*r))-mdphi_dr*(a*a+(-2+r)*r)*(real_orig[8]*a-real_orig[5]*(a*a+(-2+r)*r)+real_orig[2]*(a*a+r*r))*sinmdphi+(-2*imag_orig[8]*a*(-1+r)+imag_orig_dr[8]*a*(a*a+(-2+r)*r)+2*imag_orig[2]*r*(a*a+(-2+r)*r)-imag_orig_dr[5]*pow(a*a+(-2+r)*r,2)-2*imag_orig[2]*(-1+r)*(a*a+r*r)+imag_orig_dr[2]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi))/pow(a*a+(-2+r)*r,2));
imag_conv_dr[5]=(factor*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(real_orig[8]*a-real_orig[5]*(a*a+(-2+r)*r)+real_orig[2]*(a*a+r*r))-cosmdphi*(-2*imag_orig[8]*a*(-1+r)+imag_orig_dr[8]*a*(a*a+(-2+r)*r)+2*imag_orig[2]*r*(a*a+(-2+r)*r)-imag_orig_dr[5]*pow(a*a+(-2+r)*r,2)-2*imag_orig[2]*(-1+r)*(a*a+r*r)+imag_orig_dr[2]*(a*a+(-2+r)*r)*(a*a+r*r))+mdphi_dr*(a*a+(-2+r)*r)*(imag_orig[8]*a-imag_orig[5]*(a*a+(-2+r)*r)+imag_orig[2]*(a*a+r*r))*sinmdphi+(-2*real_orig[8]*a*(-1+r)+real_orig_dr[8]*a*(a*a+(-2+r)*r)+2*real_orig[2]*r*(a*a+(-2+r)*r)-real_orig_dr[5]*pow(a*a+(-2+r)*r,2)-2*real_orig[2]*(-1+r)*(a*a+r*r)+real_orig_dr[2]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi))/pow(a*a+(-2+r)*r,2);
real_conv_dr[6]=-((factor*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(imag_orig[9]*a-imag_orig[6]*(a*a+(-2+r)*r)+imag_orig[3]*(a*a+r*r))+cosmdphi*(-2*real_orig[9]*a*(-1+r)+real_orig_dr[9]*a*(a*a+(-2+r)*r)+2*real_orig[3]*r*(a*a+(-2+r)*r)-real_orig_dr[6]*pow(a*a+(-2+r)*r,2)-2*real_orig[3]*(-1+r)*(a*a+r*r)+real_orig_dr[3]*(a*a+(-2+r)*r)*(a*a+r*r))-mdphi_dr*(a*a+(-2+r)*r)*(real_orig[9]*a-real_orig[6]*(a*a+(-2+r)*r)+real_orig[3]*(a*a+r*r))*sinmdphi+(-2*imag_orig[9]*a*(-1+r)+imag_orig_dr[9]*a*(a*a+(-2+r)*r)+2*imag_orig[3]*r*(a*a+(-2+r)*r)-imag_orig_dr[6]*pow(a*a+(-2+r)*r,2)-2*imag_orig[3]*(-1+r)*(a*a+r*r)+imag_orig_dr[3]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi))/(pow(a*a+(-2+r)*r,2)*sinth));
imag_conv_dr[6]=(factor*(mdphi_dr*cosmdphi*(a*a+(-2+r)*r)*(real_orig[9]*a-real_orig[6]*(a*a+(-2+r)*r)+real_orig[3]*(a*a+r*r))-cosmdphi*(-2*imag_orig[9]*a*(-1+r)+imag_orig_dr[9]*a*(a*a+(-2+r)*r)+2*imag_orig[3]*r*(a*a+(-2+r)*r)-imag_orig_dr[6]*pow(a*a+(-2+r)*r,2)-2*imag_orig[3]*(-1+r)*(a*a+r*r)+imag_orig_dr[3]*(a*a+(-2+r)*r)*(a*a+r*r))+mdphi_dr*(a*a+(-2+r)*r)*(imag_orig[9]*a-imag_orig[6]*(a*a+(-2+r)*r)+imag_orig[3]*(a*a+r*r))*sinmdphi+(-2*real_orig[9]*a*(-1+r)+real_orig_dr[9]*a*(a*a+(-2+r)*r)+2*real_orig[3]*r*(a*a+(-2+r)*r)-real_orig_dr[6]*pow(a*a+(-2+r)*r,2)-2*real_orig[3]*(-1+r)*(a*a+r*r)+real_orig_dr[3]*(a*a+(-2+r)*r)*(a*a+r*r))*sinmdphi))/(pow(a*a+(-2+r)*r,2)*sinth);
real_conv_dr[7]=(factor*(cosmdphi*(-real_orig[7]+real_orig_dr[7]*r)+(-imag_orig[7]+imag_orig_dr[7]*r)*sinmdphi+mdphi_dr*r*(imag_orig[7]*cosmdphi-real_orig[7]*sinmdphi)))/(r*r);
imag_conv_dr[7]=-((factor*(cosmdphi*(imag_orig[7]-imag_orig_dr[7]*r)+(-real_orig[7]+real_orig_dr[7]*r)*sinmdphi+mdphi_dr*r*(real_orig[7]*cosmdphi+imag_orig[7]*sinmdphi)))/(r*r));
real_conv_dr[8]=(factor*(cosmdphi*(-real_orig[8]+real_orig_dr[8]*r)+(-imag_orig[8]+imag_orig_dr[8]*r)*sinmdphi+mdphi_dr*r*(imag_orig[8]*cosmdphi-real_orig[8]*sinmdphi)))/(r*r*sinth);
imag_conv_dr[8]=-((factor*(cosmdphi*(imag_orig[8]-imag_orig_dr[8]*r)+(-real_orig[8]+real_orig_dr[8]*r)*sinmdphi+mdphi_dr*r*(real_orig[8]*cosmdphi+imag_orig[8]*sinmdphi)))/(r*r*sinth));
real_conv_dr[9]=(factor*(cosmdphi*(-real_orig[9]+real_orig_dr[9]*r)+(-imag_orig[9]+imag_orig_dr[9]*r)*sinmdphi+mdphi_dr*r*(imag_orig[9]*cosmdphi-real_orig[9]*sinmdphi)))/(r*r*sinth*sinth);
imag_conv_dr[9]=-((factor*(cosmdphi*(imag_orig[9]-imag_orig_dr[9]*r)+(-real_orig[9]+real_orig_dr[9]*r)*sinmdphi+mdphi_dr*r*(real_orig[9]*cosmdphi+imag_orig[9]*sinmdphi)))/(r*r*sinth*sinth));
}

}  // namespace GrSelfForce::detail
// NOLINTEND
