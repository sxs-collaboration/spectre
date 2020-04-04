// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <pup.h>  // IWYU pragma: keep

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace Elasticity {
namespace Solutions {

double alpha(const double k, const double shear_modulus,
             const double beam_width, const double applied_force) {
  return 1 / (4. * M_PI * shear_modulus) * exp(-square(k * beam_width / 2.)) *
         applied_force;
}

double integrand_displacement_w(const double k, const double w, const double z,
                                const double shear_modulus,
                                const double poisson_ratio,
                                const double beam_width,
                                const double applied_force) noexcept {
  return (-1 + 2. * poisson_ratio + k * z) * exp(-k * z) *
         gsl_sf_bessel_J1(w * k) *
         alpha(k, shear_modulus, beam_width, applied_force);
}

double integrand_displacement_z(const double k, const double w, const double z,
                                const double shear_modulus,
                                const double poisson_ratio,
                                const double beam_width,
                                const double applied_force) noexcept {
  return (2 - 2. * poisson_ratio + k * z) * exp(-k * z) *
         gsl_sf_bessel_J0(w * k) *
         alpha(k, shear_modulus, beam_width, applied_force);
}

double integrand_displacement_w_dw(const double k, const double w,
                                   const double z, const double shear_modulus,
                                   const double poisson_ratio,
                                   const double beam_width,
                                   const double applied_force) noexcept {
  return (-1 + 2. * poisson_ratio + k * z) * exp(-k * z) * k *
         (gsl_sf_bessel_J0(w * k) - gsl_sf_bessel_Jn(2, w * k)) / 2. *
         alpha(k, shear_modulus, beam_width, applied_force);
}

double integrand_displacement_w_dz(const double k, const double w,
                                   const double z, const double shear_modulus,
                                   const double poisson_ratio,
                                   const double beam_width,
                                   const double applied_force) noexcept {
  return (2 - 2. * poisson_ratio - k * z) * k * exp(-k * z) *
         gsl_sf_bessel_J1(w * k) *
         alpha(k, shear_modulus, beam_width, applied_force);
}

double integrand_displacement_z_dw(const double k, const double w,
                                   const double z, const double shear_modulus,
                                   const double poisson_ratio,
                                   const double beam_width,
                                   const double applied_force) noexcept {
  return (2 - 2. * poisson_ratio + k * z) * exp(-k * z) * k *
         (-gsl_sf_bessel_J1(w * k)) *
         alpha(k, shear_modulus, beam_width, applied_force);
}

double integrand_displacement_z_dz(const double k, const double w,
                                   const double z, const double shear_modulus,
                                   const double poisson_ratio,
                                   const double beam_width,
                                   const double applied_force) noexcept {
  return (-1 + 2. * poisson_ratio - k * z) * k * exp(-k * z) *
         gsl_sf_bessel_J0(w * k) *
         alpha(k, shear_modulus, beam_width, applied_force);
}

HalfSpaceMirror::HalfSpaceMirror(
    double beam_width, double applied_force,
    constitutive_relation_type constitutive_relation, size_t no_intervals,
    double absolute_tolerance) noexcept
    : beam_width_(beam_width),
      applied_force_(applied_force),
      constitutive_relation_(std::move(constitutive_relation)),
      no_intervals_(no_intervals),
      absolute_tolerance_(absolute_tolerance) {}

HalfSpaceMirror::HalfSpaceMirror(double beam_width, double applied_force,
                                 double bulk_modulus, double shear_modulus,
                                 size_t no_intervals,
                                 double absolute_tolerance) noexcept
    : beam_width_(beam_width),
      applied_force_(applied_force),
      constitutive_relation_(
          Elasticity::ConstitutiveRelations::IsotropicHomogeneous<dim>{
              bulk_modulus, shear_modulus}),
      no_intervals_(no_intervals),
      absolute_tolerance_(absolute_tolerance) {}

tuples::TaggedTuple<Tags::Displacement<dim>> HalfSpaceMirror::variables(
    const tnsr::I<DataVector, dim>& x,
    tmpl::list<Tags::Displacement<dim>> /*meta*/) const noexcept {
  const double shear_modulus = constitutive_relation_.shear_modulus();
  const double poisson_ratio = constitutive_relation_.poisson_ratio();
  auto result = make_with_value<tnsr::I<DataVector, 3>>(x, 0.);
  const auto w = sqrt(square(get<0>(x)) + square(get<1>(x))) +
                 std::numeric_limits<double>::epsilon();
  double cos_phi;
  double sin_phi;

  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{no_intervals_};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();
  double z;
  double r;
  for (size_t i = 0; i < num_points; i++) {
    z = get<2>(x)[i];
    r = w[i];
    auto result_w = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_w(k, r, z, shear_modulus, poisson_ratio,
                                          this->beam_width_,
                                          this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);

    if (w[i] <= 1e-13) {
      ASSERT(
          result_w <= 1e-13,
          "The Jacobian is singular at the origin, the field value has to be "
          "zero there for the transformation to make sense");
      cos_phi = 0.;
      sin_phi = 0.;
    }

    else {
      cos_phi = get<0>(x)[i] / w[i];
      sin_phi = get<1>(x)[i] / w[i];
    }

    get<0>(result)[i] = result_w * cos_phi;
    get<1>(result)[i] = result_w * sin_phi;

    auto result_z = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_z(k, r, z, shear_modulus, poisson_ratio,
                                          this->beam_width_,
                                          this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);
    get<2>(result)[i] = result_z;
  }
  return {std::move(result)};
}

tuples::TaggedTuple<Tags::Strain<dim>> HalfSpaceMirror::variables(
    const tnsr::I<DataVector, dim>& x,
    tmpl::list<Tags::Strain<dim>> /*meta*/) const noexcept {
  const double shear_modulus = constitutive_relation_.shear_modulus();
  const double poisson_ratio = constitutive_relation_.poisson_ratio();
  auto result = make_with_value<tnsr::ii<DataVector, 3>>(x, 0.);
  const auto w = sqrt(square(get<0>(x)) + square(get<1>(x))) +
                 std::numeric_limits<double>::epsilon();
  double cos_phi;
  double sin_phi;

  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{no_intervals_};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();
  double r;
  double z;
  for (size_t i = 0; i < num_points; i++) {
    r = w[i];
    z = get<2>(x)[i];
    auto result_xiw_dw = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_w_dw(k, r, z, shear_modulus,
                                             poisson_ratio, this->beam_width_,
                                             this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);

    auto result_xiw_dz = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_w_dz(k, r, z, shear_modulus,
                                             poisson_ratio, this->beam_width_,
                                             this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);

    auto result_xiz_dw = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_z_dw(k, r, z, shear_modulus,
                                             poisson_ratio, this->beam_width_,
                                             this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);

    auto result_xiz_dz = integration(
        [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
          return integrand_displacement_z_dz(k, r, z, shear_modulus,
                                             poisson_ratio, this->beam_width_,
                                             this->applied_force_);
        },
        lower_boundary, absolute_tolerance_);

    double xi_w_over_w;
    if (w[i] <= 1e-13) {
      cos_phi = 0.;
      sin_phi = 0.;
      xi_w_over_w = result_xiw_dw;
    } else {
      cos_phi = get<0>(x)[i] / w[i];
      sin_phi = get<1>(x)[i] / w[i];
      auto result_w = integration(
          [&r, &z, &shear_modulus, &poisson_ratio, this](const double k) {
            return integrand_displacement_w(k, r, z, shear_modulus,
                                            poisson_ratio, this->beam_width_,
                                            this->applied_force_);
          },
          lower_boundary, absolute_tolerance_);

      xi_w_over_w = result_w / w[i];
    }
    get<0, 0>(result)[i] =
        xi_w_over_w + square(cos_phi) * (result_xiw_dw - xi_w_over_w);
    get<0, 1>(result)[i] = cos_phi * sin_phi * (result_xiw_dw - xi_w_over_w);
    get<0, 2>(result)[i] = cos_phi * (result_xiw_dz + result_xiz_dw) / 2.;
    get<1, 1>(result)[i] =
        xi_w_over_w + square(sin_phi) * (result_xiw_dw - xi_w_over_w);
    get<1, 2>(result)[i] = sin_phi * (result_xiw_dz + result_xiz_dw) / 2.;
    get<1, 0>(result)[i] = get<0, 1>(result)[i];
    get<2, 0>(result)[i] = get<0, 2>(result)[i];
    get<2, 1>(result)[i] = get<1, 2>(result)[i];
    get<2, 2>(result)[i] = result_xiz_dz;
  }
  return {std::move(result)};
}

tuples::TaggedTuple<::Tags::FixedSource<Tags::Displacement<dim>>>
HalfSpaceMirror::variables(
    const tnsr::I<DataVector, dim>& x,
    tmpl::list<::Tags::FixedSource<Tags::Displacement<dim>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataVector, dim>>(x, 0.)};
}

tuples::TaggedTuple<::Tags::Initial<Tags::Displacement<dim>>>
HalfSpaceMirror::variables(
    const tnsr::I<DataVector, dim>& x,
    tmpl::list<::Tags::Initial<Tags::Displacement<dim>>> /*meta*/) const
    noexcept {
  return {make_with_value<tnsr::I<DataVector, dim>>(x, 0.)};
}

void HalfSpaceMirror::pup(PUP::er& p) noexcept {
  p | beam_width_;
  p | applied_force_;
  p | constitutive_relation_;
}

bool operator==(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept {
  return lhs.beam_width_ == rhs.beam_width_ and
         lhs.applied_force_ == rhs.applied_force_ and
         lhs.constitutive_relation_ == rhs.constitutive_relation_;
}

bool operator!=(const HalfSpaceMirror& lhs,
                const HalfSpaceMirror& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Solutions
}  // namespace Elasticity
