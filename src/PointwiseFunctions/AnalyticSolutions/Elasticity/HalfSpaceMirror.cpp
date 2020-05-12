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

static constexpr size_t dim = HalfSpaceMirror::dim;

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
  const double lame_parameter = constitutive_relation_.lame_parameter();
  auto result = make_with_value<tnsr::I<DataVector, dim>>(x, 0.);
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
        [&r, &z, &shear_modulus, &lame_parameter, this](const double k) {
          return applied_force_ / (2. * shear_modulus) *
                 gsl_sf_bessel_J1(k * r) * exp(-k * z) *
                 (1 -
                  (lame_parameter + 2. * shear_modulus) /
                      (lame_parameter + shear_modulus) +
                  k * z) *
                 1 / (2. * M_PI) * exp(-square(k * beam_width_ / 2.));
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
        [&r, &z, &shear_modulus, &lame_parameter, this](const double k) {
          return applied_force_ / (2. * shear_modulus) *
                 gsl_sf_bessel_J0(k * r) * exp(-k * z) *
                 (1 + shear_modulus / (lame_parameter + shear_modulus) +
                  k * z) *
                 1 / (2. * M_PI) * exp(-square(k * beam_width_ / 2.));
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
  const double lame_parameter = constitutive_relation_.lame_parameter();
  auto strain = make_with_value<tnsr::ii<DataVector, dim>>(x, 0.);
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
    auto theta = integration(
        [&r, &z, &shear_modulus, &lame_parameter, this](const double k) {
          return applied_force_ / (2. * shear_modulus) * k *
                 gsl_sf_bessel_J0(k * r) *
                 (-2. * shear_modulus / (lame_parameter + shear_modulus)) *
                 exp(-k * z) * 1 / (2. * M_PI) *
                 exp(-square(k * beam_width_ / 2.));
        },
        lower_boundary, absolute_tolerance_);

    auto strain_rz = integration(
        [&r, &z, &shear_modulus, this](const double k) {
          return -applied_force_ / (2. * shear_modulus) * k *
                 gsl_sf_bessel_J1(k * r) * (k * z) * exp(-k * z) * 1 /
                 (2. * M_PI) * exp(-square(k * beam_width_ / 2.));
        },
        lower_boundary, absolute_tolerance_);

    auto strain_zz = integration(
        [&r, &z, &shear_modulus, &lame_parameter, this](const double k) {
          return applied_force_ / (2. * shear_modulus) * k *
                 gsl_sf_bessel_J0(k * r) * exp(-k * z) *
                 (-shear_modulus / (lame_parameter + shear_modulus) - k * z) *
                 1 / (2. * M_PI) * exp(-square(k * beam_width_ / 2.));
        },
        lower_boundary, absolute_tolerance_);

    double strain_rr;
    double strain_pp;
    if (w[i] <= 1e-13) {
      cos_phi = 0.;
      sin_phi = 0.;
      strain_rr = 0.5 * (theta - strain_zz);
      strain_pp = strain_rr;
    } else {
      auto displacement_w = integration(
          [&r, &z, &shear_modulus, &lame_parameter, this](const double k) {
            return applied_force_ / (2. * shear_modulus) *
                   gsl_sf_bessel_J1(k * r) * exp(-k * z) *
                   (1 -
                    (lame_parameter + 2. * shear_modulus) /
                        (lame_parameter + shear_modulus) +
                    k * z) *
                   1 / (2. * M_PI) * exp(-square(k * beam_width_ / 2.));
          },
          lower_boundary, absolute_tolerance_);

      cos_phi = get<0>(x)[i] / w[i];
      sin_phi = get<1>(x)[i] / w[i];
      strain_pp = displacement_w / w[i];
      strain_rr = theta - strain_pp - strain_zz;
    }
    get<0, 0>(strain)[i] =
        strain_pp + square(cos_phi) * (strain_rr - strain_pp);
    get<0, 1>(strain)[i] = cos_phi * sin_phi * (strain_rr - strain_pp);
    get<0, 2>(strain)[i] = cos_phi * strain_rz;
    get<1, 1>(strain)[i] =
        strain_pp + square(sin_phi) * (strain_rr - strain_pp);
    get<1, 2>(strain)[i] = sin_phi * strain_rz;
    get<1, 0>(strain)[i] = get<0, 1>(strain)[i];
    get<2, 0>(strain)[i] = get<0, 2>(strain)[i];
    get<2, 1>(strain)[i] = get<1, 2>(strain)[i];
    get<2, 2>(strain)[i] = strain_zz;
  }
  return {std::move(strain)};
}

Scalar<DataVector> HalfSpaceMirror::pointwise_isotropic_energy(
    const tnsr::I<DataVector, dim>& x) const noexcept {
  const double shear_modulus = constitutive_relation_.shear_modulus();
  const double lame_parameter = constitutive_relation_.lame_parameter();
  Scalar<DataVector> pointwise_potential =
      make_with_value<Scalar<DataVector>>(x, 0.);
  auto strain = get<::Elasticity::Tags::Strain<dim>>(
      variables(x, tmpl::list<::Elasticity::Tags::Strain<dim>>{}));
  double strain_square;
  double theta;
  const size_t num_points = get<0>(x).size();
  for (size_t i = 0; i < num_points; i++) {
    theta = 0;
    theta += get<0, 0>(strain)[i];
    theta += get<1, 1>(strain)[i];
    theta += get<2, 2>(strain)[i];
    strain_square = 0;
    strain_square += square(get<0, 0>(strain)[i]);
    strain_square += square(get<1, 1>(strain)[i]);
    strain_square += square(get<2, 2>(strain)[i]);
    strain_square += 2. * square(get<0, 1>(strain)[i]);
    strain_square += 2. * square(get<0, 2>(strain)[i]);
    strain_square += 2. * square(get<1, 2>(strain)[i]);
    get(pointwise_potential)[i] =
        lame_parameter / 2. * square(theta) + shear_modulus * strain_square;
  }
  return pointwise_potential;
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
