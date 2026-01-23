// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbit.hpp"

#include <complex>
#include <cstddef>
#include <effsource_gr.hpp>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbitCoeffs.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/AnalyticData/CircularOrbitConvertEffsource.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Gsl.hpp"

namespace GrSelfForce::AnalyticData {

CircularOrbit::CircularOrbit(const double black_hole_mass,
                             const double black_hole_spin,
                             const double orbital_radius,
                             const int m_mode_number)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number) {}

CircularOrbit::CircularOrbit(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> CircularOrbit::puncture_position() const {
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
      r_0 - r_plus, M, black_hole_spin_);
  return tnsr::I<double, 2>{{{r_star, M_PI_2}}};
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar, Tags::GammaTheta>
CircularOrbit::variables(const tnsr::I<DataVector, 2>& x,
                         tmpl::list<Tags::Alpha, Tags::Beta, Tags::GammaRstar,
                                    Tags::GammaTheta> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
  const DataVector cos_theta = cos(theta);
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - square(cos_theta);
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::GammaRstar,
                      Tags::GammaTheta>
      result{};
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma_rstar = get<Tags::GammaRstar>(result);
  auto& gamma_theta = get<Tags::GammaTheta>(result);
  const size_t num_points = r.size();
  get(alpha) = delta / r_sq_plus_a_sq_sq;
  for (size_t i = 0; i < beta.size(); ++i) {
    beta[i] = ComplexDataVector{num_points, 0.};
    gamma_rstar[i] = ComplexDataVector{num_points, 0.};
    gamma_theta[i] = ComplexDataVector{num_points, 0.};
  }
  const ComplexDataVector temp1 =
      1. / r * std::complex<double>(0., 2. * a * m_mode_number_);
  // tt, tr, ttheta, tphi, rr, rtheta, rphi, theta theta, theta phi, phi phi
  std::array<std::array<double, 10>, 10> Areal{};
  std::array<std::array<double, 10>, 10> Aimag{};
  std::array<std::array<double, 10>, 10> Breal{};
  std::array<std::array<double, 10>, 10> Bimag{};
  std::array<std::array<double, 10>, 10> Creal{};
  std::array<std::array<double, 10>, 10> Cimag{};
  for (size_t i = 0; i < r.size(); i++) {
    detail::getAreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Areal);
    detail::getAimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Aimag);
    detail::getBreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Breal);
    detail::getBimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Bimag);
    detail::getCreal(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Creal);
    detail::getCimag(m_mode_number_, a, m_mode_number_ * omega, r[i], theta[i],
                     Cimag);
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
    for (size_t a1 = 0; a1 < 4; ++a1) {
      for (size_t b = 0; b <= a1; ++b) {
        const size_t matrix_i =
            tnsr::aa<ComplexDataVector, 3>::get_storage_index(
                std::array<size_t, 2>{{a1, b}});
        for (size_t c = 0; c < 4; ++c) {
          for (size_t d = 0; d <= c; ++d) {
            const size_t matrix_j =
                tnsr::aa<ComplexDataVector, 3>::get_storage_index(
                    std::array<size_t, 2>{{c, d}});
            gamma_rstar.get(a1, b, c, d)[i] =
                Areal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Aimag[matrix_i][matrix_j];
            gamma_theta.get(a1, b, c, d)[i] =
                Breal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Bimag[matrix_i][matrix_j];
            beta.get(a1, b, c, d)[i] =
                Creal[matrix_i][matrix_j] +
                std::complex<double>(0., 1.) * Cimag[matrix_i][matrix_j];
          }
        }
      }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  }
  for (size_t i = 0; i < beta.size(); ++i) {
    beta[i] *= -1.;
    gamma_rstar[i] *= -1.;
    gamma_theta[i] *= -1. / get(alpha);
  }
  return result;
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/) {
  tuples::TaggedTuple<Tags::MMode> result{};
  auto& field = get<Tags::MMode>(result);
  for (size_t i = 0; i < field.size(); ++i) {
    field[i] = ComplexDataVector{get<0>(x).size(), 0.};
  }
  return result;
}

// Fixed sources
tuples::TaggedTuple<
    ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
    ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
    Tags::BoyerLindquistRadius>
CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<
        ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
        ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
        Tags::BoyerLindquistRadius> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_0 = orbital_radius_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  {
    // Initialize effsource
    effsource_init(M, a);
    coordinate xp{};
    xp.t = 0;
    xp.r = r_0;
    xp.theta = M_PI_2;
    xp.phi = 0;
    // Circular equatorial orbit, as given in the EffectiveSource example
    const double e = ((r_0 - 2.0 * M) * sqrt(M * r_0) + a * M) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    const double l = (M * (a * a + r_0 * r_0 - 2.0 * a * sqrt(M * r_0))) /
                     (sqrt(M * r_0) * sqrt(r_0 * r_0 - 3.0 * M * r_0 +
                                           2.0 * a * sqrt(M * r_0)));
    effsource_set_particle(&xp, e, l, 0.);
  }
  const auto& r_star = get<0>(x);
  const auto& theta = get<1>(x);
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector delta_phi = m_mode_number_ * a / (r_plus - r_minus) *
                               log((r - r_plus) / (r - r_minus));
  const ComplexDataVector rotation =
      cos(delta_phi) - std::complex<double>(0., 1.) * sin(delta_phi);
  tuples::TaggedTuple<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>
      result{};
  get(get<Tags::BoyerLindquistRadius>(result)) = r;
  const size_t num_points = get<0>(x).size();
  tnsr::aa<ComplexDataVector, 3>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  tnsr::aa<ComplexDataVector, 3>& singular_field =
      get<Tags::SingularField>(result);
  for (size_t i = 0; i < singular_field.size(); i++) {
    effective_source[i].destructive_resize(num_points);
    singular_field[i].destructive_resize(num_points);
  }
  auto& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  for (size_t i = 0; i < deriv_singular_field.size(); i++) {
    deriv_singular_field[i].destructive_resize(num_points);
  }
  {
    // Call into effsource
    coordinate x_i{};
    std::array<double, 10> hS_re{};
    std::array<double, 10> hS_im{};
    std::array<double, 10> hS_conv_re{};
    std::array<double, 10> hS_conv_im{};
    std::array<double, 10> dhS_dr_re{};
    std::array<double, 10> dhS_dr_im{};
    std::array<double, 10> dhS_dth_re{};
    std::array<double, 10> dhS_dth_im{};
    std::array<double, 10> dhS_dph_re{};
    std::array<double, 10> dhS_dph_im{};
    std::array<double, 10> dhS_dt_re{};
    std::array<double, 10> dhS_dt_im{};
    std::array<double, 10> dhS_drstar_re_conv{};
    std::array<double, 10> dhS_drstar_im_conv{};
    std::array<double, 10> dhS_dth_re_conv{};
    std::array<double, 10> dhS_dth_im_conv{};
    std::array<double, 10> src_re{};
    std::array<double, 10> src_im{};
    std::array<double, 10> src_conv_re{};
    std::array<double, 10> src_conv_im{};
    for (size_t i = 0; i < get<0>(x).size(); ++i) {
      x_i.t = 0;
      x_i.r = r[i];
      x_i.theta = theta[i];
      x_i.phi = 0;
      effsource_calc_m(m_mode_number_, &x_i, hS_re.data(), hS_im.data(),
                       dhS_dr_re.data(), dhS_dr_im.data(), dhS_dth_re.data(),
                       dhS_dth_im.data(), dhS_dph_re.data(), dhS_dph_im.data(),
                       dhS_dt_re.data(), dhS_dt_im.data(), src_re.data(),
                       src_im.data());
      detail::convert_effsource_psi(m_mode_number_, a, r[i], theta[i], hS_re,
                                    hS_im, hS_conv_re, hS_conv_im);
      detail::convert_effsource_dpsidtheta(m_mode_number_, a, r[i], theta[i],
                                           hS_re, hS_im, dhS_dth_re, dhS_dth_im,
                                           dhS_dth_re_conv, dhS_dth_im_conv);
      detail::convert_effsource_dpsidrstar(
          m_mode_number_, a, r[i], theta[i], hS_re, hS_im, dhS_dr_re, dhS_dr_im,
          dhS_drstar_re_conv, dhS_drstar_im_conv);
      detail::convert_effsource_Seff(m_mode_number_, a, r[i], theta[i], src_re,
                                     src_im, src_conv_re, src_conv_im);
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      for (size_t a1 = 0; a1 < 4; ++a1) {
        for (size_t b = 0; b <= a1; ++b) {
          const size_t comp = tnsr::aa<ComplexDataVector, 3>::get_storage_index(
              std::array<size_t, 2>{{a1, b}});
          effective_source.get(a1, b)[i] =
              -src_conv_re[comp] -
              std::complex<double>(0., 1.) * src_conv_im[comp];
          singular_field.get(a1, b)[i] =
              hS_conv_re[comp] +
              std::complex<double>(0., 1.) * hS_conv_im[comp];
          deriv_singular_field.get(0, a1, b)[i] =
              dhS_drstar_re_conv[comp] +
              std::complex<double>(0., 1.) * dhS_drstar_im_conv[comp];
          deriv_singular_field.get(1, a1, b)[i] =
              dhS_dth_re_conv[comp] +
              std::complex<double>(0., 1.) * dhS_dth_im_conv[comp];
        }
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
    }
  }
  return result;
}

void CircularOrbit::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | black_hole_mass_;
  p | black_hole_spin_;
  p | orbital_radius_;
  p | m_mode_number_;
}

bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_;
}

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID CircularOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace GrSelfForce::AnalyticData
