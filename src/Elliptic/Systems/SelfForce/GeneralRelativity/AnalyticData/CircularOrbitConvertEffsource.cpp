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

}  // namespace GrSelfForce::detail
// NOLINTEND
