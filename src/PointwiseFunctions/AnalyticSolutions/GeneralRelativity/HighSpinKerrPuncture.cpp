// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HighSpinKerrPuncture.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace gr::Solutions {

HighSpinKerrPuncture::HighSpinKerrPuncture(const double mass,
                                           const double dimensionless_spin,
                                           const Options::Context& context)
    : mass_(mass), dimensionless_spin_(dimensionless_spin) {
  if (mass <= 0.) {
    PARSE_ERROR(context,
                "Black hole mass must be positive, but given " << mass_);
  }
  if (abs(dimensionless_spin) >= 1.) {
    PARSE_ERROR(context,
                "The dimensionless spin must satisfy |chi| < 1 strictly, "
                "because gamma_rr diverges at the throat r = r_+/4 as "
                "|chi| -> 1 (the infinite proper throat of extremal Kerr), but "
                "given "
                    << dimensionless_spin_);
  }
}

HighSpinKerrPuncture::HighSpinKerrPuncture(CkMigrateMessage* /*msg*/) {}

void HighSpinKerrPuncture::pup(PUP::er& p) {
  p | mass_;
  p | dimensionless_spin_;
}

template <typename DataType>
HighSpinKerrPuncture::IntermediateVars<DataType>::IntermediateVars(
    const double mass_in, const double dimensionless_spin,
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x)
    : r(sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)))),
      z(get<2>(x)),
      mass(mass_in),
      spin_a(dimensionless_spin * mass_in),
      r_plus(mass_in + mass_in * sqrt((1. - dimensionless_spin) *
                                      (1. + dimensionless_spin))),
      r_minus(square(spin_a) / r_plus) {
  ASSERT(min(r) > 0.,
         "HighSpinKerrPuncture: every point must have coordinate radius "
         "r > 0. The puncture sits at the origin and must not coincide with a "
         "grid point. min(r) = "
             << min(r));

  one_over_r = 1. / r;

  const double a_squared = square(spin_a);
  const double r_plus_over_four = 0.25 * r_plus;

  // Boyer-Lindquist radius and its r-derivative, Eq. (11) of Liu, Etienne &
  // Shapiro (LES), PRD 80, 121503 (2009):
  //   r_BL = r (1 + r_+/4r)^2 = r + r_+/2 + r_+^2/(16 r),
  //   r_BL' = 1 - r_+^2/(16 r^2) = (r - r_+/4)(r + r_+/4)/r^2.
  r_bl = r + 0.5 * r_plus + square(r_plus) / 16. * one_over_r;

  // r - r_+/4, the throat-vanishing factor.
  r_minus_r_plus_over_four = r - r_plus_over_four;

  d_r_bl_d_r =
      r_minus_r_plus_over_four * (r + r_plus_over_four) * square(one_over_r);

  const DataType r_bl_minus_r_plus =
      square(r_minus_r_plus_over_four) * one_over_r;

  const double r_plus_minus_r_minus =
      2. * mass_in *
      sqrt((1. - dimensionless_spin) * (1. + dimensionless_spin));
  r_bl_minus_r_minus = r_bl_minus_r_plus + r_plus_minus_r_minus;

  const DataType z_squared_over_r_squared = square(z) * square(one_over_r);
  const DataType one_minus_z_squared_over_r_squared =
      1. - z_squared_over_r_squared;

  // Boyer-Lindquist scalars.
  //   Sigma = r_BL^2 + a^2 z^2/r^2
  //   Delta = (r_BL - r_+)(r_BL - r_-)
  //   A = (r_BL^2 + a^2)^2 - Delta a^2 (1 - z^2/r^2)
  //   G = 3 r_BL^4 + 2 a^2 r_BL^2 - a^4 - a^2 (r_BL^2 - a^2)(1 - z^2/r^2)
  sigma = square(r_bl) + a_squared * z_squared_over_r_squared;
  delta = r_bl_minus_r_plus * r_bl_minus_r_minus;
  const DataType r_bl_squared_plus_a_squared = square(r_bl) + a_squared;
  a_capital = square(r_bl_squared_plus_a_squared) -
              delta * a_squared * one_minus_z_squared_over_r_squared;
  g_capital = 3. * pow<4>(r_bl) + 2. * a_squared * square(r_bl) -
              square(a_squared) -
              a_squared * (square(r_bl) - a_squared) *
                  one_minus_z_squared_over_r_squared;

  // (r, z) partials of the Boyer-Lindquist scalars.
  const DataType d_delta_d_r = 2. * (r_bl - mass) * d_r_bl_d_r;
  d_sigma_d_r =
      2. * r_bl * d_r_bl_d_r - 2. * a_squared * square(z) * pow<3>(one_over_r);
  d_sigma_d_z = 2. * a_squared * z * square(one_over_r);
  d_a_capital_d_r =
      4. * r_bl * r_bl_squared_plus_a_squared * d_r_bl_d_r -
      a_squared * d_delta_d_r * one_minus_z_squared_over_r_squared -
      2. * a_squared * delta * square(z) * pow<3>(one_over_r);
  d_a_capital_d_z = 2. * a_squared * delta * z * square(one_over_r);
  d_g_capital_d_r =
      (12. * pow<3>(r_bl) + 4. * a_squared * r_bl) * d_r_bl_d_r -
      a_squared *
          (2. * r_bl * d_r_bl_d_r * one_minus_z_squared_over_r_squared +
           2. * (square(r_bl) - a_squared) * square(z) * pow<3>(one_over_r));
  d_g_capital_d_z =
      2. * a_squared * (square(r_bl) - a_squared) * z * square(one_over_r);

  // Cartesian coefficient functions of the assembled tensors and their (r, z)
  // partials.

  // c_delta = Sigma / r^2
  c_delta = sigma * square(one_over_r);
  d_c_delta_d_r =
      d_sigma_d_r * square(one_over_r) - 2. * sigma * pow<3>(one_over_r);
  d_c_delta_d_z = d_sigma_d_z * square(one_over_r);

  // c_n = Sigma r_- / (r^2 (r_BL - r_-))
  const DataType one_over_r_bl_minus_r_minus = 1. / r_bl_minus_r_minus;
  c_n = sigma * r_minus * square(one_over_r) * one_over_r_bl_minus_r_minus;
  d_c_n_d_r =
      r_minus * one_over_r_bl_minus_r_minus *
      (d_sigma_d_r * square(one_over_r) - 2. * sigma * pow<3>(one_over_r) -
       sigma * square(one_over_r) * d_r_bl_d_r * one_over_r_bl_minus_r_minus);
  d_c_n_d_z =
      r_minus * d_sigma_d_z * square(one_over_r) * one_over_r_bl_minus_r_minus;

  // c_lambda = a^2 (Sigma + 2 M r_BL) / (Sigma r^4)
  const DataType sigma_plus_two_m_r_bl = sigma + 2. * mass * r_bl;
  const DataType d_sigma_plus_two_m_r_bl_d_r =
      d_sigma_d_r + 2. * mass * d_r_bl_d_r;
  const DataType one_over_sigma = 1. / sigma;
  c_lambda =
      a_squared * sigma_plus_two_m_r_bl * one_over_sigma * pow<4>(one_over_r);
  d_c_lambda_d_r =
      a_squared *
      (d_sigma_plus_two_m_r_bl_d_r * one_over_sigma * pow<4>(one_over_r) -
       sigma_plus_two_m_r_bl * d_sigma_d_r * square(one_over_sigma) *
           pow<4>(one_over_r) -
       4. * sigma_plus_two_m_r_bl * one_over_sigma * pow<5>(one_over_r));
  d_c_lambda_d_z =
      a_squared * (d_sigma_d_z * one_over_sigma * pow<4>(one_over_r) -
                   sigma_plus_two_m_r_bl * d_sigma_d_z *
                       square(one_over_sigma) * pow<4>(one_over_r));

  // c_{n lambda} = M a G sqrt(r_BL) / (Sigma sqrt(A Sigma) r^3 sqrt(r_BL-r_-))
  //             = M a G r_BL^{1/2} A^{-1/2} Sigma^{-3/2} r^{-3}
  //               (r_BL - r_-)^{-1/2}.
  const DataType one_over_a_capital = 1. / a_capital;
  c_n_lambda = mass * spin_a * g_capital * sqrt(r_bl) *
               sqrt(one_over_a_capital) * one_over_sigma *
               sqrt(one_over_sigma) * pow<3>(one_over_r) *
               sqrt(one_over_r_bl_minus_r_minus);
  // Derivatives by the product/quotient/power rule (log-derivative form; every
  // factor of c_{n lambda} is nonzero for r > 0, |chi| < 1).
  const DataType c_n_lambda_log_deriv_r =
      d_g_capital_d_r / g_capital + 0.5 * d_r_bl_d_r / r_bl -
      0.5 * d_a_capital_d_r * one_over_a_capital -
      1.5 * d_sigma_d_r * one_over_sigma - 3. * one_over_r -
      0.5 * d_r_bl_d_r * one_over_r_bl_minus_r_minus;
  const DataType c_n_lambda_log_deriv_z =
      d_g_capital_d_z / g_capital - 0.5 * d_a_capital_d_z * one_over_a_capital -
      1.5 * d_sigma_d_z * one_over_sigma;
  d_c_n_lambda_d_r = c_n_lambda * c_n_lambda_log_deriv_r;
  d_c_n_lambda_d_z = c_n_lambda * c_n_lambda_log_deriv_z;

  // c_{mu lambda} = (r - r_+/4) z c_mu_lambda_base with
  //   c_mu_lambda_base = -2 a^3 M r_BL (r_BL - r_-)^{1/2} A^{-1/2}
  //                       Sigma^{-3/2} r^{-13/2}.
  c_mu_lambda_base = -2. * pow<3>(spin_a) * mass * r_bl *
                     sqrt(r_bl_minus_r_minus) * sqrt(one_over_a_capital) *
                     one_over_sigma * sqrt(one_over_sigma) *
                     pow<6>(one_over_r) * sqrt(one_over_r);
  c_mu_lambda = r_minus_r_plus_over_four * z * c_mu_lambda_base;
  const DataType c_mu_lambda_base_log_deriv_r =
      d_r_bl_d_r / r_bl + 0.5 * d_r_bl_d_r * one_over_r_bl_minus_r_minus -
      0.5 * d_a_capital_d_r * one_over_a_capital -
      1.5 * d_sigma_d_r * one_over_sigma - 6.5 * one_over_r;
  const DataType c_mu_lambda_base_log_deriv_z =
      -0.5 * d_a_capital_d_z * one_over_a_capital -
      1.5 * d_sigma_d_z * one_over_sigma;
  const DataType d_c_mu_lambda_base_d_r =
      c_mu_lambda_base * c_mu_lambda_base_log_deriv_r;
  const DataType d_c_mu_lambda_base_d_z =
      c_mu_lambda_base * c_mu_lambda_base_log_deriv_z;
  // c_{mu lambda},r = z [c_mu_lambda_base + (r - r_+/4) c_mu_lambda_base,r]
  d_c_mu_lambda_d_r = z * (c_mu_lambda_base +
                           r_minus_r_plus_over_four * d_c_mu_lambda_base_d_r);
  // c_{mu lambda},z = (r - r_+/4) [c_mu_lambda_base + z c_mu_lambda_base,z]
  d_c_mu_lambda_d_z = r_minus_r_plus_over_four *
                      (c_mu_lambda_base + z * d_c_mu_lambda_base_d_z);

  // Lapse factor g = sqrt((r_BL - r_-) Sigma / (r A)). The class's lapse is
  // the signed alpha = (r - r_+/4) g; its absolute value is the LES lapse
  // sqrt(Delta Sigma / A).
  const DataType w_lapse =
      r_bl_minus_r_minus * sigma * one_over_r * one_over_a_capital;
  g_lapse = sqrt(w_lapse);
  const DataType d_w_lapse_d_r =
      d_r_bl_d_r * sigma * one_over_r * one_over_a_capital +
      r_bl_minus_r_minus * d_sigma_d_r * one_over_r * one_over_a_capital -
      r_bl_minus_r_minus * sigma * square(one_over_r) * one_over_a_capital -
      r_bl_minus_r_minus * sigma * one_over_r * d_a_capital_d_r *
          square(one_over_a_capital);
  const DataType d_w_lapse_d_z =
      r_bl_minus_r_minus * d_sigma_d_z * one_over_r * one_over_a_capital -
      r_bl_minus_r_minus * sigma * one_over_r * d_a_capital_d_z *
          square(one_over_a_capital);
  d_g_lapse_d_r = 0.5 * d_w_lapse_d_r / g_lapse;
  d_g_lapse_d_z = 0.5 * d_w_lapse_d_z / g_lapse;

  // c_beta = -2 M a r_BL / A
  c_beta = -2. * mass * spin_a * r_bl * one_over_a_capital;
  d_c_beta_d_r = -2. * mass * spin_a *
                 (d_r_bl_d_r * one_over_a_capital -
                  r_bl * d_a_capital_d_r * square(one_over_a_capital));
  d_c_beta_d_z =
      2. * mass * spin_a * r_bl * d_a_capital_d_z * square(one_over_a_capital);
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>> {
  // alpha = (r - r_+/4) g, the signed (Killing) lapse: smooth everywhere
  // except the puncture, negative on the inner sheet r < r_+/4. The
  // nonnegative lapse of Liu, Etienne, and Shapiro Eq. (6),
  // sqrt(Delta Sigma / A), is its absolute value.
  return {Scalar<DataType>{vars.r_minus_r_plus_over_four * vars.g_lapse}};
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> {
  return {make_with_value<Scalar<DataType>>(vars.r, 0.)};
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivLapse<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivLapse<DataType>> {
  // d_k alpha = [g + (r - r_+/4) g,r] n_k + (r - r_+/4) g,z delta_{k3},
  // the product rule on the signed lapse alpha = (r - r_+/4) g.
  const DataType radial_part =
      vars.g_lapse + vars.r_minus_r_plus_over_four * vars.d_g_lapse_d_r;
  const DataType z_part = vars.r_minus_r_plus_over_four * vars.d_g_lapse_d_z;
  auto d_lapse =
      make_with_value<tnsr::i<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                      0.);
  for (size_t k = 0; k < volume_dim; ++k) {
    d_lapse.get(k) = radial_part * x.get(k) * vars.one_over_r;
  }
  get<2>(d_lapse) += z_part;
  return d_lapse;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::Shift<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::Shift<DataType, volume_dim>> {
  // beta^i = c_beta lambda^i with lambda^i = (-y, x, 0).
  auto shift = make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(
      vars.r, 0.);
  get<0>(shift) = -vars.c_beta * get<1>(x);
  get<1>(shift) = vars.c_beta * get<0>(x);
  return shift;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> /*meta*/)
    const
    -> tuples::TaggedTuple<::Tags::dt<gr::Tags::Shift<DataType, volume_dim>>> {
  return {make_with_value<tnsr::I<DataType, volume_dim, Frame::Inertial>>(
      vars.r, 0.)};
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivShift<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivShift<DataType>> {
  // d_k beta^j = (d_k c_beta) lambda^j + c_beta (d_k lambda^j),
  // with d_k c_beta = c_beta,r n_k + c_beta,z delta_{k3},
  // lambda^j = (-y, x, 0), and d_k lambda^j = epsilon_{j3k}:
  //   d_k lambda^0 = -delta_{1k}, d_k lambda^1 = delta_{0k}, d_k lambda^2 = 0.
  auto d_shift =
      make_with_value<tnsr::iJ<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                       0.);
  const std::array<DataType, volume_dim> lambda{
      {-get<1>(x), get<0>(x), make_with_value<DataType>(vars.r, 0.)}};
  for (size_t k = 0; k < volume_dim; ++k) {
    DataType d_k_c_beta = vars.d_c_beta_d_r * x.get(k) * vars.one_over_r;
    if (k == 2) {
      d_k_c_beta += vars.d_c_beta_d_z;
    }
    for (size_t j = 0; j < volume_dim; ++j) {
      d_shift.get(k, j) = d_k_c_beta * gsl::at(lambda, j);
    }
  }
  // c_beta (d_k lambda^j) contributions, with d_k lambda^0 = -delta_{1k},
  // d_k lambda^1 = delta_{0k}, d_k lambda^2 = 0.
  get<1, 0>(d_shift) -= vars.c_beta;
  get<0, 1>(d_shift) += vars.c_beta;
  return d_shift;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SpatialMetric<DataType, volume_dim>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SpatialMetric<DataType, volume_dim>> {
  // gamma_{ij} = c_delta delta_{ij} + c_n n_i n_j + c_lambda lambda_i lambda_j.
  auto spatial_metric =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                       0.);
  const std::array<DataType, volume_dim> lambda{
      {-get<1>(x), get<0>(x), make_with_value<DataType>(vars.r, 0.)}};
  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = i; j < volume_dim; ++j) {
      spatial_metric.get(i, j) =
          vars.c_n * x.get(i) * x.get(j) * square(vars.one_over_r) +
          vars.c_lambda * gsl::at(lambda, i) * gsl::at(lambda, j);
      if (i == j) {
        spatial_metric.get(i, j) += vars.c_delta;
      }
    }
  }
  return spatial_metric;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> /*meta*/)
    const -> tuples::TaggedTuple<
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, volume_dim>>> {
  return {make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(
      vars.r, 0.)};
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<DerivSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<DerivSpatialMetric<DataType>> {
  // d_k gamma_{ij} = (d_k c_delta) delta_{ij}
  //   + (d_k c_n) n_i n_j + c_n d_k(n_i n_j)
  //   + (d_k c_lambda) lambda_i lambda_j + c_lambda d_k(lambda_i lambda_j),
  // with d_k c_F = c_F,r n_k + c_F,z delta_{k3},
  //   d_k n_i = (delta_{ik} - n_i n_k)/r, d_k lambda_i = epsilon_{i3k}.
  auto d_spatial_metric =
      make_with_value<tnsr::ijj<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                        0.);
  const std::array<DataType, volume_dim> lambda{
      {-get<1>(x), get<0>(x), make_with_value<DataType>(vars.r, 0.)}};
  for (size_t k = 0; k < volume_dim; ++k) {
    const DataType n_k = x.get(k) * vars.one_over_r;
    DataType d_k_c_delta = vars.d_c_delta_d_r * n_k;
    DataType d_k_c_n = vars.d_c_n_d_r * n_k;
    DataType d_k_c_lambda = vars.d_c_lambda_d_r * n_k;
    if (k == 2) {
      d_k_c_delta += vars.d_c_delta_d_z;
      d_k_c_n += vars.d_c_n_d_z;
      d_k_c_lambda += vars.d_c_lambda_d_z;
    }
    for (size_t i = 0; i < volume_dim; ++i) {
      const DataType n_i = x.get(i) * vars.one_over_r;
      for (size_t j = i; j < volume_dim; ++j) {
        const DataType n_j = x.get(j) * vars.one_over_r;
        // d_k(n_i n_j) = (d_k n_i) n_j + n_i (d_k n_j)
        //   with d_k n_i = (delta_{ik} - n_i n_k)/r.
        const DataType d_k_n_i =
            (((i == k) ? 1. : 0.) - n_i * n_k) * vars.one_over_r;
        const DataType d_k_n_j =
            (((j == k) ? 1. : 0.) - n_j * n_k) * vars.one_over_r;
        // d_k(lambda_i lambda_j) = (d_k lambda_i) lambda_j
        //   + lambda_i (d_k lambda_j), with d_k lambda_0 = -delta_{1k},
        //   d_k lambda_1 = delta_{0k}, d_k lambda_2 = 0.
        const double d_k_lambda_i =
            (i == 0 and k == 1) ? -1. : ((i == 1 and k == 0) ? 1. : 0.);
        const double d_k_lambda_j =
            (j == 0 and k == 1) ? -1. : ((j == 1 and k == 0) ? 1. : 0.);
        d_spatial_metric.get(k, i, j) =
            d_k_c_n * n_i * n_j + vars.c_n * (d_k_n_i * n_j + n_i * d_k_n_j) +
            d_k_c_lambda * gsl::at(lambda, i) * gsl::at(lambda, j) +
            vars.c_lambda * (d_k_lambda_i * gsl::at(lambda, j) +
                             gsl::at(lambda, i) * d_k_lambda_j);
        if (i == j) {
          d_spatial_metric.get(k, i, j) += d_k_c_delta;
        }
      }
    }
  }
  return d_spatial_metric;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& /*x*/,
    const double /*t*/, const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
    -> tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> {
  // sqrt(det gamma) = r^{-3} sqrt(Sigma r_BL A / (r_BL - r_-)).
  return {Scalar<DataType>{
      pow<3>(vars.one_over_r) *
      sqrt(vars.sigma * vars.r_bl * vars.a_capital / vars.r_bl_minus_r_minus)}};
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> /*meta*/)
    const
    -> tuples::TaggedTuple<gr::Tags::ExtrinsicCurvature<DataType, volume_dim>> {
  // K_{ij} = c_{n lambda} (n_i lambda_j + n_j lambda_i)
  //        + c_{mu lambda} (mu_i lambda_j + mu_j lambda_i),
  // with n_i = x_i/r, lambda_i = (-y, x, 0),
  //   mu_i = (z x, z y, -(x^2 + y^2)) = z x_i - r^2 delta_{i3}.
  auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                       0.);
  const std::array<DataType, volume_dim> lambda{
      {-get<1>(x), get<0>(x), make_with_value<DataType>(vars.r, 0.)}};
  const DataType r_squared = square(vars.r);
  const std::array<DataType, volume_dim> mu{
      {vars.z * get<0>(x), vars.z * get<1>(x), vars.z * get<2>(x) - r_squared}};
  for (size_t i = 0; i < volume_dim; ++i) {
    const DataType n_i = x.get(i) * vars.one_over_r;
    for (size_t j = i; j < volume_dim; ++j) {
      const DataType n_j = x.get(j) * vars.one_over_r;
      extrinsic_curvature.get(i, j) =
          vars.c_n_lambda *
              (n_i * gsl::at(lambda, j) + n_j * gsl::at(lambda, i)) +
          vars.c_mu_lambda * (gsl::at(mu, i) * gsl::at(lambda, j) +
                              gsl::at(mu, j) * gsl::at(lambda, i));
    }
  }
  return extrinsic_curvature;
}

template <typename DataType>
auto HighSpinKerrPuncture::variables(
    const tnsr::I<DataType, volume_dim, Frame::Inertial>& x, const double /*t*/,
    const IntermediateVars<DataType>& vars,
    tmpl::list<gr::Tags::InverseSpatialMetric<DataType, volume_dim>> /*meta*/)
    const -> tuples::TaggedTuple<
        gr::Tags::InverseSpatialMetric<DataType, volume_dim>> {
  // gamma^{ij} = (1/c_delta) delta^{ij}
  //   - c_n / (c_delta (c_delta + c_n)) n^i n^j
  //   - c_lambda / (c_delta (c_delta + varpi^2 c_lambda)) lambda^i lambda^j,
  // with varpi^2 = x^2 + y^2 = |lambda|^2.
  auto inverse_spatial_metric =
      make_with_value<tnsr::II<DataType, volume_dim, Frame::Inertial>>(vars.r,
                                                                       0.);
  const std::array<DataType, volume_dim> lambda{
      {-get<1>(x), get<0>(x), make_with_value<DataType>(vars.r, 0.)}};
  const DataType varpi_squared = square(get<0>(x)) + square(get<1>(x));
  const DataType one_over_c_delta = 1. / vars.c_delta;
  const DataType n_coefficient =
      -vars.c_n * one_over_c_delta / (vars.c_delta + vars.c_n);
  const DataType lambda_coefficient =
      -vars.c_lambda * one_over_c_delta /
      (vars.c_delta + varpi_squared * vars.c_lambda);
  for (size_t i = 0; i < volume_dim; ++i) {
    for (size_t j = i; j < volume_dim; ++j) {
      inverse_spatial_metric.get(i, j) =
          n_coefficient * x.get(i) * x.get(j) * square(vars.one_over_r) +
          lambda_coefficient * gsl::at(lambda, i) * gsl::at(lambda, j);
      if (i == j) {
        inverse_spatial_metric.get(i, j) += one_over_c_delta;
      }
    }
  }
  return inverse_spatial_metric;
}

bool operator==(const HighSpinKerrPuncture& lhs,
                const HighSpinKerrPuncture& rhs) {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin();
}

bool operator!=(const HighSpinKerrPuncture& lhs,
                const HighSpinKerrPuncture& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>::               \
      IntermediateVars(                                                        \
          const double mass_in, const double dimensionless_spin,               \
          const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& x);          \
  template tuples::TaggedTuple<gr::Tags::Lapse<DTYPE(data)>>                   \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data), Frame::Inertial>& /*x*/,           \
      const double /*t*/,                                                      \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::Lapse<DTYPE(data)>> /*meta*/) const;                \
  template tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>>       \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DTYPE(data)>>> /*meta*/) const;    \
  template tuples::TaggedTuple<::Tags::deriv<                                  \
      gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, Frame::Inertial>> \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Lapse<DTYPE(data)>, tmpl::size_t<DIM(data)>, \
                        Frame::Inertial>> /*meta*/) const;                     \
  template tuples::TaggedTuple<gr::Tags::Shift<DTYPE(data), DIM(data)>>        \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::Shift<DTYPE(data), DIM(data)>> /*meta*/) const;     \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>>                     \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::dt<gr::Tags::Shift<DTYPE(data), DIM(data)>>> /*meta*/)       \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,                   \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::Shift<DTYPE(data), DIM(data)>,               \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>                         \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      ::Tags::dt<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>>             \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<::Tags::dt<                                                   \
          gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>>> /*meta*/) const;   \
  template tuples::TaggedTuple<                                                \
      ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,           \
                    tmpl::size_t<DIM(data)>, Frame::Inertial>>                 \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          ::Tags::deriv<gr::Tags::SpatialMetric<DTYPE(data), DIM(data)>,       \
                        tmpl::size_t<DIM(data)>, Frame::Inertial>> /*meta*/)   \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>>                  \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          gr::Tags::InverseSpatialMetric<DTYPE(data), DIM(data)>> /*meta*/)    \
      const;                                                                   \
  template tuples::TaggedTuple<                                                \
      gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>>                    \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double /*t*/,            \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<                                                              \
          gr::Tags::ExtrinsicCurvature<DTYPE(data), DIM(data)>> /*meta*/)      \
      const;                                                                   \
  template tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>>    \
  HighSpinKerrPuncture::variables(                                             \
      const tnsr::I<DTYPE(data), DIM(data)>& /*x*/, const double /*t*/,        \
      const HighSpinKerrPuncture::IntermediateVars<DTYPE(data)>& vars,         \
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DTYPE(data)>> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (3), (double, DataVector))

#undef INSTANTIATE
#undef DTYPE
#undef DIM
}  // namespace gr::Solutions
