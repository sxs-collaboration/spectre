// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gsl/gsl_sf_bessel.h>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace Elasticity::Solutions::detail {

namespace {
double displacement_r_integrand(const double k, const double r, const double z,
                                const double beam_width,
                                const double modulus_term_r) noexcept {
  return gsl_sf_bessel_J1(k * r) * exp(-k * z - square(0.5 * k * beam_width)) *
         (modulus_term_r + k * z);
}
}  // namespace

template <typename DataType>
void HalfSpaceMirrorVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> displacement,
    const gsl::not_null<Cache*> /*cache*/, Tags::Displacement<3> /*meta*/) const
    noexcept {
  const double shear_modulus = constitutive_relation.shear_modulus();
  const double lame_parameter = constitutive_relation.lame_parameter();
  const auto radius = sqrt(square(get<0>(x)) + square(get<1>(x)));

  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{integration_intervals};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();

  const double prefactor = 0.25 / (shear_modulus * M_PI);
  const double modulus_term_r = 1. - (lame_parameter + 2. * shear_modulus) /
                                         (lame_parameter + shear_modulus);
  const double modulus_term_z =
      1. + shear_modulus / (lame_parameter + shear_modulus);
  for (size_t i = 0; i < num_points; i++) {
    const double z = get<2>(x)[i];
    const double r = radius[i];
    try {
      if (not equal_within_roundoff(r, 0.)) {
        const double displacement_r =
            prefactor *
            integration(
                [&r, &z, &modulus_term_r, this](const double k) noexcept {
                  return displacement_r_integrand(k, r, z, beam_width,
                                                  modulus_term_r);
                },
                lower_boundary, absolute_tolerance, relative_tolerance);
        // projection on cartesian grid
        get<0>(*displacement)[i] = get<0>(x)[i] / r * displacement_r;
        get<1>(*displacement)[i] = get<1>(x)[i] / r * displacement_r;
      } else {
        get<0>(*displacement)[i] = 0.;
        get<1>(*displacement)[i] = 0.;
      }
      const double displacement_z =
          prefactor *
          integration(
              [&r, &z, &modulus_term_z, this](const double k) noexcept {
                return gsl_sf_bessel_J0(k * r) *
                       exp(-k * z - square(0.5 * k * beam_width)) *
                       (modulus_term_z + k * z);
              },
              lower_boundary, absolute_tolerance, relative_tolerance);
      get<2>(*displacement)[i] = displacement_z;
    } catch (convergence_error& error) {
      ERROR("The numerical integral failed at r="
            << r << ", z=" << z << " (" << error.what()
            << "). Try to increase 'IntegrationIntervals' or make the domain "
               "smaller.\n");
    }
  }
}

template <typename DataType>
void HalfSpaceMirrorVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> strain,
    const gsl::not_null<Cache*> /*cache*/, Tags::Strain<3> /*meta*/) const
    noexcept {
  const double shear_modulus = constitutive_relation.shear_modulus();
  const double lame_parameter = constitutive_relation.lame_parameter();
  const auto radius = sqrt(square(get<0>(x)) + square(get<1>(x)));
  const integration::GslQuadAdaptive<
      integration::GslIntegralType::UpperBoundaryInfinite>
      integration{integration_intervals};
  const double lower_boundary = 0.;
  const size_t num_points = get<0>(x).size();

  const double prefactor = 0.25 / (shear_modulus * M_PI);
  const double modulus_term_trace =
      -2. * shear_modulus / (lame_parameter + shear_modulus);
  const double modulus_term_zz =
      shear_modulus / (lame_parameter + shear_modulus);
  const double modulus_term_r = 1. - (lame_parameter + 2. * shear_modulus) /
                                         (lame_parameter + shear_modulus);
  for (size_t i = 0; i < num_points; i++) {
    const double r = radius[i];
    const double z = get<2>(x)[i];
    try {
      const double trace_term =
          prefactor *
          integration(
              [&r, &z, &modulus_term_trace, this](const double k) noexcept {
                return k * gsl_sf_bessel_J0(k * r) *
                       exp(-k * z - square(0.5 * k * beam_width)) *
                       modulus_term_trace;
              },
              lower_boundary, absolute_tolerance, relative_tolerance);

      const double strain_zz =
          -prefactor *
          integration(
              [&r, &z, &modulus_term_zz, this](const double k) noexcept {
                return k * gsl_sf_bessel_J0(k * r) *
                       exp(-k * z - square(0.5 * k * beam_width)) *
                       (modulus_term_zz + k * z);
              },
              lower_boundary, absolute_tolerance, relative_tolerance);

      if (not equal_within_roundoff(r, 0)) {
        const double strain_rz =
            -prefactor *
            integration(
                [&r, &z, this](const double k) noexcept {
                  return k * gsl_sf_bessel_J1(k * r) *
                         exp(-k * z - square(0.5 * k * beam_width)) * (k * z);
                },
                lower_boundary, absolute_tolerance, relative_tolerance);

        const double displacement_r =
            prefactor *
            integration(
                [&r, &z, &modulus_term_r, this](const double k) noexcept {
                  return displacement_r_integrand(k, r, z, beam_width,
                                                  modulus_term_r);
                },
                lower_boundary, absolute_tolerance, relative_tolerance);
        const double cos_phi = get<0>(x)[i] / r;
        const double sin_phi = get<1>(x)[i] / r;
        const double strain_pp = displacement_r / r;
        const double strain_rr = trace_term - strain_pp - strain_zz;
        get<0, 0>(*strain)[i] =
            strain_pp + square(cos_phi) * (strain_rr - strain_pp);
        get<0, 1>(*strain)[i] = cos_phi * sin_phi * (strain_rr - strain_pp);
        get<0, 2>(*strain)[i] = cos_phi * strain_rz;
        get<1, 1>(*strain)[i] =
            strain_pp + square(sin_phi) * (strain_rr - strain_pp);
        get<1, 2>(*strain)[i] = sin_phi * strain_rz;
        get<2, 2>(*strain)[i] = strain_zz;
      } else {
        get<0, 0>(*strain)[i] = 0.5 * (trace_term - strain_zz);
        get<0, 1>(*strain)[i] = 0.;
        get<0, 2>(*strain)[i] = 0.;
        get<1, 1>(*strain)[i] = get<0, 0>(*strain)[i];
        get<1, 2>(*strain)[i] = 0.;
        get<2, 2>(*strain)[i] = strain_zz;
      }
    } catch (convergence_error& error) {
      ERROR("The numerical integral failed at r="
            << r << ", z=" << z << " (" << error.what()
            << "). Try to increase 'IntegrationIntervals' or make the domain "
               "smaller.\n");
    }
  }
}

template <typename DataType>
void HalfSpaceMirrorVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_for_displacement,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::Displacement<3>> /*meta*/) const noexcept {
  std::fill(fixed_source_for_displacement->begin(),
            fixed_source_for_displacement->end(), 0.);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template class HalfSpaceMirrorVariables<DTYPE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))

#undef DTYPE
#undef INSTANTIATE

}  // namespace Elasticity::Solutions::detail
/// \endcond
