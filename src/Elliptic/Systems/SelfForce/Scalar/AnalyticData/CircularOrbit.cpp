// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"

#include <complex>
#include <cstddef>
#include <effsource.hpp>
#include <gsl/gsl_errno.h>
#include <utility>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TortoiseCoordinates.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ScalarSelfForce::AnalyticData {

namespace {
std::pair<DataVector, DataVector> boost_function_and_deriv(
    const DataVector& r_star, const std::array<double, 4>& transition_points) {
  return {
      smoothstep<1>(transition_points[0], transition_points[1], r_star) +
          smoothstep<1>(transition_points[2], transition_points[3], r_star) -
          1.0,
      smoothstep_deriv<1>(transition_points[0], transition_points[1], r_star) +
          smoothstep_deriv<1>(transition_points[2], transition_points[3],
                              r_star)};
}
}  // namespace

CircularOrbit::CircularOrbit(const double black_hole_mass,
                             const double black_hole_spin,
                             const double orbital_radius,
                             const int m_mode_number,
                             const std::optional<std::array<double, 4>>
                                 hyperboloidal_slicing_transitions,
                             const bool impose_equatorial_symmetry)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number),
      hyperboloidal_slicing_transitions_(hyperboloidal_slicing_transitions),
      impose_equatorial_symmetry_(impose_equatorial_symmetry) {}

CircularOrbit::CircularOrbit(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> CircularOrbit::puncture_position() const {
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
      r_0 - r_plus, M, black_hole_spin_);
  return tnsr::I<double, 2>{{{r_star, 0.}}};
}

// Background
tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma>
CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x,
    tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma> /*meta*/) const {
  const double a = black_hole_spin_ * black_hole_mass_;
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_minus = M * (1. - sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  const double omega = 1. / (a + sqrt(cube(r_0) / M));
  const auto& r_star = get<0>(x);
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
  } else {
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - cos_theta_sq;
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma> result{};
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma = get<Tags::Gamma>(result);
  get(alpha) = delta / r_sq_plus_a_sq_sq;
  const ComplexDataVector temp1 =
      1. / r * std::complex<double>(0., 2. * a * m_mode_number_);
  get(beta) = (-square(m_mode_number_ * omega) * sigma_squared +
               4. * a * square(m_mode_number_) * omega * M * r +
               delta * (m_mode_number_ * (m_mode_number_ + 1) +
                        2. * M / r * (1. - square(a) / M / r) + temp1)) /
              r_sq_plus_a_sq_sq;
  get<0>(gamma) =
      -1. / r_sq_plus_a_sq * std::complex<double>(0., 2. * a * m_mode_number_) +
      2. * square(a) * get(alpha) / r;
  get<1>(gamma) = 2. * m_mode_number_ * cos_theta_or_sq * get(alpha);
  if (impose_equatorial_symmetry_) {
    get<1>(gamma) += sin_theta_squared * get(alpha);
    get<1>(gamma) *= 2.0;
  }
  get(alpha) *= sin_theta_squared;
  if (impose_equatorial_symmetry_) {
    get(alpha) *= 4. * cos_theta_sq;
  }
  // Hyperboloidal slicing
  if (hyperboloidal_slicing_transitions_.has_value()) {
    const auto [H, dH] = boost_function_and_deriv(
        r_star, hyperboloidal_slicing_transitions_.value());
    const double k = m_mode_number_ * omega;
    get(beta) += std::complex<double>(0., -k) * dH + square(k) * square(H) +
                 std::complex<double>(0., k) * get<0>(gamma) * H;
    get<0>(gamma) += std::complex<double>(0., -2. * k) * H;
  }
  return result;
}

// Initial guess
tuples::TaggedTuple<Tags::MMode> CircularOrbit::variables(
    const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/) {
  tuples::TaggedTuple<Tags::MMode> result{};
  auto& field = get<Tags::MMode>(result);
  get(field) = ComplexDataVector{get<0>(x).size(), 0.};
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
  if (hyperboloidal_slicing_transitions_.has_value() and
      (min(r_star) < (*hyperboloidal_slicing_transitions_)[1] or
       max(r_star) > (*hyperboloidal_slicing_transitions_)[2])) {
    ERROR(
        "The effective source is only valid where no hyperboloidal slicing is "
        "applied, which is in the r_* range ["
        << (*hyperboloidal_slicing_transitions_)[1] << ", "
        << (*hyperboloidal_slicing_transitions_)[2]
        << "], but was requested in the range [" << min(r_star) << ", "
        << max(r_star) << "]");
  }
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta;
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta = sqrt(cos_theta_or_sq);
  } else {
    // NOLINTNEXTLINE
    cos_theta.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector r_minus_r_plus =
      gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(r_star, M,
                                                            black_hole_spin_);
  const DataVector r = r_minus_r_plus + r_plus;
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_sq = 1. - cos_theta_sq;
  const DataVector sin_theta = sqrt(sin_theta_sq);
  const DataVector sin_theta_pow_m = integer_pow(sin_theta, m_mode_number_);
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
  Scalar<ComplexDataVector>& effective_source =
      get<::Tags::FixedSource<Tags::MMode>>(result);
  get(effective_source).destructive_resize(num_points);
  Scalar<ComplexDataVector>& singular_field = get<Tags::SingularField>(result);
  get(singular_field).destructive_resize(num_points);
  tnsr::i<ComplexDataVector, 2>& deriv_singular_field =
      get<::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>>(
          result);
  get<0>(deriv_singular_field).destructive_resize(num_points);
  get<1>(deriv_singular_field).destructive_resize(num_points);
  {
    // Call into effsource
    coordinate x_i{};
    std::array<double, 2> PhiS{};
    std::array<double, 8> dPhiS_dx{};
    std::array<double, 20> d2PhiS_dx2{};
    std::array<double, 2> src{};
    for (size_t i = 0; i < num_points; ++i) {
      x_i.t = 0;
      x_i.r = r[i];
      x_i.theta = acos(cos_theta[i]);
      x_i.phi = 0;
      effsource_calc_m(m_mode_number_, &x_i, PhiS.data(), dPhiS_dx.data(),
                       d2PhiS_dx2.data(), src.data());
      get(effective_source)[i] = src[0] + std::complex<double>(0., 1.) * src[1];
      get(singular_field)[i] = PhiS[0] + std::complex<double>(0., 1.) * PhiS[1];
      get<0>(deriv_singular_field)[i] =
          dPhiS_dx[2] + std::complex<double>(0., 1.) * dPhiS_dx[3];
      get<1>(deriv_singular_field)[i] =
          dPhiS_dx[4] + std::complex<double>(0., 1.) * dPhiS_dx[5];
    }
  }
  // Rotate the source by delta_phi and multiply by r / 2 pi
  get(effective_source) *= rotation * 0.5 * r / M_PI;
  // Factor Delta * (r^2 + a^2 cos^2(theta)) / Sigma^2
  // Factor Sigma^2 / (r^2 + a^2)^2 from first-order formulation
  // Factor 1/sin(theta)^m from change of variables
  get(effective_source) *= delta * (square(r) + square(a) * cos_theta_sq) /
                           r_sq_plus_a_sq_sq / sin_theta_pow_m;
  get(singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  get<0>(deriv_singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  get<0>(deriv_singular_field) +=
      get(singular_field) / r - std::complex<double>(0., a * m_mode_number_) /
                                    delta * get(singular_field);
  get<0>(deriv_singular_field) *= delta / r_sq_plus_a_sq;
  get<1>(deriv_singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  get<1>(deriv_singular_field) /= -sin_theta;
  if (impose_equatorial_symmetry_) {
    get<1>(deriv_singular_field) /= 2. * cos_theta;
  }
  {
    ComplexDataVector add_term =
        m_mode_number_ * get(singular_field) / sin_theta_sq;
    if (impose_equatorial_symmetry_) {
      add_term *= 0.5;
    } else {
      add_term *= cos_theta;
    }
    get<1>(deriv_singular_field) += add_term;
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
  p | hyperboloidal_slicing_transitions_;
  p | impose_equatorial_symmetry_;
}

bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_ and
         lhs.hyperboloidal_slicing_transitions_ ==
             rhs.hyperboloidal_slicing_transitions_ and
         lhs.impose_equatorial_symmetry_ == rhs.impose_equatorial_symmetry_;
}

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID CircularOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AnalyticData
