// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"

#include <complex>
#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

/// \cond

void SphericalMetricData::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<
        gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>> /*meta*/)
    const noexcept {
  Variables<
      tmpl::list<Tags::detail::InverseCartesianToSphericalJacobian,
                 gr::Tags::SpacetimeMetric<
                     3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>>
      intermediate_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  auto& intermediate_spherical_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Spherical<::Frame::Inertial>,
                                    DataVector>>(intermediate_variables);
  auto& intermediate_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          intermediate_variables);

  spherical_metric(make_not_null(&intermediate_spherical_metric), l_max, time);
  inverse_jacobian(make_not_null(&intermediate_jacobian), l_max);

  get<0, 0>(*spacetime_metric) = get<0, 0>(intermediate_spherical_metric);
  for (size_t i = 0; i < 3; ++i) {
    spacetime_metric->get(0, i + 1) = intermediate_jacobian.get(i, 0) *
                                      get<0, 1>(intermediate_spherical_metric);
    for (size_t k = 1; k < 3; ++k) {
      spacetime_metric->get(0, i + 1) +=
          intermediate_jacobian.get(i, k) *
          intermediate_spherical_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      spacetime_metric->get(i + 1, j + 1) =
          intermediate_jacobian.get(i, 0) * intermediate_jacobian.get(j, 0) *
          get<1, 1>(intermediate_spherical_metric);
      for (size_t k = 1; k < 3; ++k) {
        spacetime_metric->get(i + 1, j + 1) +=
            (intermediate_jacobian.get(i, k) * intermediate_jacobian.get(j, 0) +
             intermediate_jacobian.get(i, 0) *
                 intermediate_jacobian.get(j, k)) *
            intermediate_spherical_metric.get(k + 1, 1);
        for (size_t l = 1; l < 3; ++l) {
          spacetime_metric->get(i + 1, j + 1) +=
              intermediate_jacobian.get(i, k) *
              intermediate_jacobian.get(j, l) *
              intermediate_spherical_metric.get(k + 1, l + 1);
        }
      }
    }
  }
}

void SphericalMetricData::variables_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<::Tags::dt<
        gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>> /*meta*/)
    const noexcept {
  Variables<
      tmpl::list<Tags::detail::InverseCartesianToSphericalJacobian,
                 ::Tags::dt<gr::Tags::SpacetimeMetric<
                     3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>>>
      intermediate_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};
  auto& intermediate_dt_spherical_metric =
      get<::Tags::dt<gr::Tags::SpacetimeMetric<
          3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>>(
          intermediate_variables);
  auto& intermediate_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          intermediate_variables);

  dt_spherical_metric(make_not_null(&intermediate_dt_spherical_metric), l_max,
                      time);
  inverse_jacobian(make_not_null(&intermediate_jacobian), l_max);

  get<0, 0>(*dt_spacetime_metric) = get<0, 0>(intermediate_dt_spherical_metric);
  for (size_t i = 0; i < 3; ++i) {
    dt_spacetime_metric->get(0, i + 1) =
        intermediate_jacobian.get(i, 0) *
        get<0, 1>(intermediate_dt_spherical_metric);
    for (size_t k = 1; k < 3; ++k) {
      dt_spacetime_metric->get(0, i + 1) +=
          intermediate_jacobian.get(i, k) *
          intermediate_dt_spherical_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      dt_spacetime_metric->get(i + 1, j + 1) =
          intermediate_jacobian.get(i, 0) * intermediate_jacobian.get(j, 0) *
          get<1, 1>(intermediate_dt_spherical_metric);
      for (size_t k = 1; k < 3; ++k) {
        dt_spacetime_metric->get(i + 1, j + 1) +=
            (intermediate_jacobian.get(i, k) * intermediate_jacobian.get(j, 0) +
             intermediate_jacobian.get(i, 0) *
                 intermediate_jacobian.get(j, k)) *
            intermediate_dt_spherical_metric.get(k + 1, 1);
        for (size_t l = 1; l < 3; ++l) {
          dt_spacetime_metric->get(i + 1, j + 1) +=
              intermediate_jacobian.get(i, k) *
              intermediate_jacobian.get(j, l) *
              intermediate_dt_spherical_metric.get(k + 1, l + 1);
        }
      }
    }
  }
}

void SphericalMetricData::variables_impl(
    const gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric,
    const size_t l_max, const double time,
    tmpl::type_<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>> /*meta*/)
    const noexcept {
  Variables<tmpl::list<
      gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
      gr::Tags::SpacetimeMetric<3, ::Frame::Spherical<::Frame::Inertial>,
                                DataVector>,
      Tags::Dr<gr::Tags::SpacetimeMetric<
          3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>,
      Tags::Dr<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      Tags::detail::InverseCartesianToSphericalJacobian,
      Tags::Dr<Tags::detail::InverseCartesianToSphericalJacobian>>>
      intermediate_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  auto& intermediate_cartesian_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          intermediate_variables);
  variables_impl(
      make_not_null(&intermediate_cartesian_metric), l_max, time,
      tmpl::type_<
          gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>{});
  auto& intermediate_spherical_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Spherical<::Frame::Inertial>,
                                    DataVector>>(intermediate_variables);
  auto& intermediate_dr_spherical_metric =
      get<Tags::Dr<gr::Tags::SpacetimeMetric<
          3, ::Frame::Spherical<::Frame::Inertial>, DataVector>>>(
          intermediate_variables);
  auto& intermediate_dr_cartesian_metric = get<
      Tags::Dr<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      intermediate_variables);
  auto& intermediate_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          intermediate_variables);
  auto& intermediate_dr_jacobian =
      get<Tags::Dr<Tags::detail::InverseCartesianToSphericalJacobian>>(
          intermediate_variables);
  spherical_metric(make_not_null(&intermediate_spherical_metric), l_max, time);
  dr_spherical_metric(make_not_null(&intermediate_dr_spherical_metric), l_max,
                      time);
  inverse_jacobian(make_not_null(&intermediate_jacobian), l_max);
  dr_inverse_jacobian(make_not_null(&intermediate_dr_jacobian), l_max);

  get<0, 0>(intermediate_dr_cartesian_metric) =
      get<0, 0>(intermediate_dr_spherical_metric);
  for (size_t i = 0; i < 3; ++i) {
    intermediate_dr_cartesian_metric.get(0, i + 1) =
        intermediate_jacobian.get(i, 0) *
            get<0, 1>(intermediate_dr_spherical_metric) +
        intermediate_dr_jacobian.get(i, 0) *
            get<0, 1>(intermediate_spherical_metric);
    for (size_t k = 1; k < 3; ++k) {
      intermediate_dr_cartesian_metric.get(0, i + 1) +=
          intermediate_jacobian.get(i, k) *
              intermediate_dr_spherical_metric.get(0, k + 1) +
          intermediate_dr_jacobian.get(i, k) *
              intermediate_spherical_metric.get(0, k + 1);
    }
    for (size_t j = i; j < 3; ++j) {
      intermediate_dr_cartesian_metric.get(i + 1, j + 1) =
          intermediate_jacobian.get(i, 0) * intermediate_jacobian.get(j, 0) *
              get<1, 1>(intermediate_dr_spherical_metric) +
          (intermediate_dr_jacobian.get(i, 0) *
               intermediate_jacobian.get(j, 0) +
           intermediate_jacobian.get(i, 0) *
               intermediate_dr_jacobian.get(j, 0)) *
              get<1, 1>(intermediate_spherical_metric);
      for (size_t k = 1; k < 3; ++k) {
        intermediate_dr_cartesian_metric.get(i + 1, j + 1) +=
            (intermediate_jacobian.get(i, k) * intermediate_jacobian.get(j, 0) +
             intermediate_jacobian.get(i, 0) *
                 intermediate_jacobian.get(j, k)) *
                intermediate_dr_spherical_metric.get(k + 1, 1) +
            (intermediate_dr_jacobian.get(i, k) *
                 intermediate_jacobian.get(j, 0) +
             intermediate_jacobian.get(i, k) *
                 intermediate_dr_jacobian.get(j, 0) +
             intermediate_dr_jacobian.get(i, 0) *
                 intermediate_jacobian.get(j, k) +
             intermediate_jacobian.get(i, 0) *
                 intermediate_dr_jacobian.get(j, k)) *
                intermediate_spherical_metric.get(1, k + 1);

        for (size_t l = 1; l < 3; ++l) {
          intermediate_dr_cartesian_metric.get(i + 1, j + 1) +=
              intermediate_jacobian.get(i, k) *
                  intermediate_jacobian.get(j, l) *
                  intermediate_dr_spherical_metric.get(k + 1, l + 1) +
              (intermediate_dr_jacobian.get(i, k) *
                   intermediate_jacobian.get(j, l) +
               intermediate_jacobian.get(i, k) *
                   intermediate_dr_jacobian.get(j, l)) *
                  intermediate_spherical_metric.get(k + 1, l + 1);
        }
      }
    }
  }

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      buffers{Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  auto& derivative_buffer =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(buffers));
  auto& pre_derivative_buffer =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(buffers));

  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      pre_derivative_buffer.data() = std::complex<double>(1.0, 0.0) *
                                     intermediate_cartesian_metric.get(a, b);
      Spectral::Swsh::angular_derivatives<
          tmpl::list<Spectral::Swsh::Tags::Eth>>(
          l_max, 1, make_not_null(&derivative_buffer), pre_derivative_buffer);
      for (size_t i = 0; i < 3; ++i) {
        d_spacetime_metric->get(i, a, b) =
            intermediate_jacobian.get(i, 0) *
                intermediate_dr_cartesian_metric.get(a, b) -
            intermediate_jacobian.get(i, 1) * real(derivative_buffer.data()) -
            intermediate_jacobian.get(i, 2) * imag(derivative_buffer.data());
      }
    }
  }
}

void SphericalMetricData::inverse_jacobian(
    const gsl::not_null<CartesianiSphericalJ*> inverse_jacobian,
    const size_t l_max) const noexcept {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    // dr/dx   dtheta/dx   dphi/dx * sin(theta)
    get<0, 0>(*inverse_jacobian)[collocation_point.offset] =
        cos(collocation_point.phi) * sin(collocation_point.theta);
    get<0, 1>(*inverse_jacobian)[collocation_point.offset] =
        cos(collocation_point.phi) * cos(collocation_point.theta) /
        extraction_radius_;
    get<0, 2>(*inverse_jacobian)[collocation_point.offset] =
        -sin(collocation_point.phi) / extraction_radius_;
    // dr/dy   dtheta/dy   dphi/dy * sin(theta)
    get<1, 0>(*inverse_jacobian)[collocation_point.offset] =
        sin(collocation_point.phi) * sin(collocation_point.theta);
    get<1, 1>(*inverse_jacobian)[collocation_point.offset] =
        cos(collocation_point.theta) * sin(collocation_point.phi) /
        extraction_radius_;
    get<1, 2>(*inverse_jacobian)[collocation_point.offset] =
        cos(collocation_point.phi) / (extraction_radius_);
    // dr/dz   dtheta/dz   dphi/dz * sin(theta)
    get<2, 0>(*inverse_jacobian)[collocation_point.offset] =
        cos(collocation_point.theta);
    get<2, 1>(*inverse_jacobian)[collocation_point.offset] =
        -sin(collocation_point.theta) / extraction_radius_;
    get<2, 2>(*inverse_jacobian)[collocation_point.offset] = 0.0;
  }
}

void SphericalMetricData::jacobian(
    const gsl::not_null<SphericaliCartesianJ*> jacobian,
    const size_t l_max) const noexcept {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    // dx/dr   dy/dr  dz/dr
    get<0, 0>(*jacobian)[collocation_point.offset] =
        sin(collocation_point.theta) * cos(collocation_point.phi);
    get<0, 1>(*jacobian)[collocation_point.offset] =
        sin(collocation_point.theta) * sin(collocation_point.phi);
    get<0, 2>(*jacobian)[collocation_point.offset] =
        cos(collocation_point.theta);
    // dx/dtheta   dy/dtheta  dz/dtheta
    get<1, 0>(*jacobian)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.theta) *
        cos(collocation_point.phi);
    get<1, 1>(*jacobian)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.theta) *
        sin(collocation_point.phi);
    get<1, 2>(*jacobian)[collocation_point.offset] =
        -extraction_radius_ * sin(collocation_point.theta);
    // (1/sin(theta)) { dx/dphi,   dy/dphi,  dz/dphi }
    get<2, 0>(*jacobian)[collocation_point.offset] =
        -extraction_radius_ * sin(collocation_point.phi);
    get<2, 1>(*jacobian)[collocation_point.offset] =
        extraction_radius_ * cos(collocation_point.phi);
    get<2, 2>(*jacobian)[collocation_point.offset] = 0.0;
  }
}

void SphericalMetricData::dr_inverse_jacobian(
    const gsl::not_null<CartesianiSphericalJ*> dr_inverse_jacobian,
    const size_t l_max) const noexcept {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    // radial derivatives of:
    // dr/dx   dtheta/dx   dphi/dx * sin(theta)
    get<0, 0>(*dr_inverse_jacobian)[collocation_point.offset] = 0.0;
    get<0, 1>(*dr_inverse_jacobian)[collocation_point.offset] =
        -cos(collocation_point.phi) * cos(collocation_point.theta) /
        square(extraction_radius_);
    get<0, 2>(*dr_inverse_jacobian)[collocation_point.offset] =
        sin(collocation_point.phi) / square(extraction_radius_);
    // radial derivatives of:
    // dr/dy   dtheta/dy   dphi/dy * sin(theta)
    get<1, 0>(*dr_inverse_jacobian)[collocation_point.offset] = 0.0;
    get<1, 1>(*dr_inverse_jacobian)[collocation_point.offset] =
        -cos(collocation_point.theta) * sin(collocation_point.phi) /
        square(extraction_radius_);
    get<1, 2>(*dr_inverse_jacobian)[collocation_point.offset] =
        -cos(collocation_point.phi) / square(extraction_radius_);
    // radial derivatives of:
    // dr/dz   dtheta/dz   dphi/dz * sin(theta)
    get<2, 0>(*dr_inverse_jacobian)[collocation_point.offset] = 0.0;
    get<2, 1>(*dr_inverse_jacobian)[collocation_point.offset] =
        sin(collocation_point.theta) / square(extraction_radius_);
    get<2, 2>(*dr_inverse_jacobian)[collocation_point.offset] = 0.0;
  }
}

void SphericalMetricData::dr_jacobian(
    const gsl::not_null<SphericaliCartesianJ*> dr_jacobian,
    const size_t l_max) noexcept {
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    // dx/dr   dy/dr  dz/dr
    get<0, 0>(*dr_jacobian)[collocation_point.offset] = 0.0;
    get<0, 1>(*dr_jacobian)[collocation_point.offset] = 0.0;
    get<0, 2>(*dr_jacobian)[collocation_point.offset] = 0.0;
    // dx/dtheta   dy/dtheta  dz/dtheta
    get<1, 0>(*dr_jacobian)[collocation_point.offset] =
        cos(collocation_point.theta) * cos(collocation_point.phi);
    get<1, 1>(*dr_jacobian)[collocation_point.offset] =
        cos(collocation_point.theta) * sin(collocation_point.phi);
    get<1, 2>(*dr_jacobian)[collocation_point.offset] =
        -sin(collocation_point.theta);
    // (1/sin(theta)) { dx/dphi,   dy/dphi,  dz/dphi }
    get<2, 0>(*dr_jacobian)[collocation_point.offset] =
        -sin(collocation_point.phi);
    get<2, 1>(*dr_jacobian)[collocation_point.offset] =
        cos(collocation_point.phi);
    get<2, 2>(*dr_jacobian)[collocation_point.offset] = 0.0;
  }
}

void SphericalMetricData::pup(PUP::er& p) noexcept { WorldtubeData::pup(p); }

/// \endcond
}  // namespace Cce::Solutions
