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
                             const bool penetrating_horizon,
                             const bool impose_equatorial_symmetry)
    : black_hole_mass_(black_hole_mass),
      black_hole_spin_(black_hole_spin),
      orbital_radius_(orbital_radius),
      m_mode_number_(m_mode_number),
      hyperboloidal_slicing_transitions_(hyperboloidal_slicing_transitions),
      penetrating_horizon_(penetrating_horizon),
      impose_equatorial_symmetry_(impose_equatorial_symmetry) {
  if (penetrating_horizon_ and
      not hyperboloidal_slicing_transitions_.has_value()) {
    ERROR(
        "Hyperboloidal slicing must be enabled when penetrating_horizon is "
        "true.");
  }
}

CircularOrbit::CircularOrbit(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

tnsr::I<double, 2> CircularOrbit::puncture_position() const {
  const double M = black_hole_mass_;
  const double r_plus = M * (1. + sqrt(1. - square(black_hole_spin_)));
  const double r_0 = orbital_radius_;
  if (penetrating_horizon_) {
    return tnsr::I<double, 2>{{{r_0, 0.}}};
  } else {
    const double r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_0 - r_plus, M, black_hole_spin_);
    return tnsr::I<double, 2>{{{r_star, 0.}}};
  }
}

double CircularOrbit::omega() const {
  const double M = black_hole_mass_;
  const double a = black_hole_spin_ * M;
  const double r_0 = orbital_radius_;
  return 1. / (a + sqrt(cube(r_0) / M));
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
  const double omega = this->omega();
  const double k = m_mode_number_ * omega;

  // Resolve coordinates
  const auto& r_star_or_r = get<0>(x);
  DataVector r;
  DataVector r_star;
  DataVector r_minus_r_plus;
  if (penetrating_horizon_) {
    // NOLINTNEXTLINE
    r.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = r - r_plus;
    r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_minus_r_plus, M, black_hole_spin_);
  } else {
    // NOLINTNEXTLINE
    r_star.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
        r_star, M, black_hole_spin_);
    r = r_minus_r_plus + r_plus;
  }
  const auto& cos_theta_or_sq = get<1>(x);
  DataVector cos_theta_sq;
  if (impose_equatorial_symmetry_) {
    // NOLINTNEXTLINE
    cos_theta_sq.set_data_ref(const_cast<DataVector*>(&cos_theta_or_sq));
  } else {
    cos_theta_sq = square(cos_theta_or_sq);
  }
  const DataVector delta = r_minus_r_plus * (r - r_minus);
  const DataVector r_sq_plus_a_sq = square(r) + square(a);
  const DataVector r_sq_plus_a_sq_sq = square(r_sq_plus_a_sq);
  const DataVector sin_theta_squared = 1. - cos_theta_sq;
  const DataVector sigma_squared =
      r_sq_plus_a_sq_sq - square(a) * delta * sin_theta_squared;
  tuples::TaggedTuple<Tags::Alpha, Tags::Beta, Tags::Gamma> result{};
  // Hyperboloidal slicing
  ComplexDataVector H;
  ComplexDataVector dH;
  if (hyperboloidal_slicing_transitions_.has_value()) {
    std::tie(H, dH) = boost_function_and_deriv(
        r_star_or_r, hyperboloidal_slicing_transitions_.value());
  } else {
    H = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
    dH = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
  }
  auto& alpha = get<Tags::Alpha>(result);
  auto& beta = get<Tags::Beta>(result);
  auto& gamma = get<Tags::Gamma>(result);
  if (penetrating_horizon_) {
    get<0>(alpha) = delta / r_sq_plus_a_sq;
    get<1>(alpha) = 1.0 / r_sq_plus_a_sq;
  } else {
    get<0>(alpha) = make_with_value<DataVector>(r_star_or_r, 1.0);
    get<1>(alpha) = delta / r_sq_plus_a_sq_sq;
  }
  get(beta) = make_with_value<ComplexDataVector>(r_star_or_r, 0.);
  for (size_t p = 0; p < get(beta).size(); ++p) {
    if (penetrating_horizon_ and equal_within_roundoff(r[p], r_plus)) {
      // The following terms are zero at the horizon. Skip them to avoid
      // division by zero.
      continue;
    }
    get(beta)[p] =
        square(k) *
            (square(H[p]) - sigma_squared[p] / r_sq_plus_a_sq_sq[p]) +
        2. * a * m_mode_number_ * k *
            (2. * M * r[p] / r_sq_plus_a_sq[p] + H[p]) / r_sq_plus_a_sq[p];
    if (penetrating_horizon_) {
      get(beta)[p] /= get<0>(alpha)[p];
    }
  }
  get(beta) +=
      get<1>(alpha) * (m_mode_number_ * (m_mode_number_ + 1) +
                       2. * M / r * (1. - square(a) / M / r) +
                       std::complex<double>(0., 2. * a * m_mode_number_) *
                           (1. + a * omega * H) / r) -
      std::complex<double>(0., k) * dH;
  get<0>(gamma) =
      2. * square(a) * delta / (r * r_sq_plus_a_sq_sq) -
      std::complex<double>(0., 2. * a * m_mode_number_) / r_sq_plus_a_sq -
      std::complex<double>(0., 2. * k) * H;
  get<1>(gamma) = 2. * m_mode_number_ * cos_theta_or_sq * get<1>(alpha);
  if (impose_equatorial_symmetry_) {
    get<1>(gamma) += sin_theta_squared * get<1>(alpha);
    get<1>(gamma) *= 2.0;
  }
 get<1>(alpha) *= sin_theta_squared;
  if (impose_equatorial_symmetry_) {
    get<1>(alpha) *= 4. * cos_theta_sq;
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
  const auto& r_star_or_r = get<0>(x);
  if (hyperboloidal_slicing_transitions_.has_value() and
      ((min(r_star_or_r) < (*hyperboloidal_slicing_transitions_)[1] and
        not equal_within_roundoff(min(r_star_or_r),
                                  (*hyperboloidal_slicing_transitions_)[1])) or
       (max(r_star_or_r) > (*hyperboloidal_slicing_transitions_)[2] and
        not equal_within_roundoff(max(r_star_or_r),
                                  (*hyperboloidal_slicing_transitions_)[2])))) {
    ERROR(
        "The effective source is only valid where no hyperboloidal slicing is "
        "applied, which is in the radial range ["
        << (*hyperboloidal_slicing_transitions_)[1] << ", "
        << (*hyperboloidal_slicing_transitions_)[2]
        << "], but was requested in the range [" << min(r_star_or_r) << ", "
        << max(r_star_or_r) << "]");
  }
  DataVector r;
  DataVector r_star;
  DataVector r_minus_r_plus;
  if (penetrating_horizon_) {
    // NOLINTNEXTLINE
    r.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = r - r_plus;
    r_star = gr::tortoise_radius_from_boyer_lindquist_minus_r_plus(
        r_minus_r_plus, M, black_hole_spin_);
  } else {
    // NOLINTNEXTLINE
    r_star.set_data_ref(const_cast<DataVector*>(&r_star_or_r));
    r_minus_r_plus = gr::boyer_lindquist_radius_minus_r_plus_from_tortoise(
        r_star, M, black_hole_spin_);
    r = r_minus_r_plus + r_plus;
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
  get(effective_source) *= (square(r) + square(a) * cos_theta_sq) /
                           (r_sq_plus_a_sq * sin_theta_pow_m);
  if (not penetrating_horizon_) {
    get(effective_source) *= delta / r_sq_plus_a_sq;
  }
  get(singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  get<0>(deriv_singular_field) *= rotation * 0.5 * r / M_PI / sin_theta_pow_m;
  get<0>(deriv_singular_field) +=
      get(singular_field) / r - std::complex<double>(0., a * m_mode_number_) /
                                    delta * get(singular_field);
  if (not penetrating_horizon_) {
    get<0>(deriv_singular_field) *= delta / r_sq_plus_a_sq;
  }
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
  p | penetrating_horizon_;
  p | impose_equatorial_symmetry_;
}

bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return lhs.black_hole_mass_ == rhs.black_hole_mass_ and
         lhs.black_hole_spin_ == rhs.black_hole_spin_ and
         lhs.orbital_radius_ == rhs.orbital_radius_ and
         lhs.m_mode_number_ == rhs.m_mode_number_ and
         lhs.hyperboloidal_slicing_transitions_ ==
             rhs.hyperboloidal_slicing_transitions_ and
         lhs.penetrating_horizon_ == rhs.penetrating_horizon_ and
         lhs.impose_equatorial_symmetry_ == rhs.impose_equatorial_symmetry_;
}

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID CircularOrbit::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AnalyticData
