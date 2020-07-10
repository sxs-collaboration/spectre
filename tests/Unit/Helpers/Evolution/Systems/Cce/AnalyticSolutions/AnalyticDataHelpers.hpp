// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Cce {
namespace Solutions {
namespace TestHelpers {

// This function determines the Bondi-Sachs scalars from a Cartesian spacetime
// metric, assuming that the metric is already in null form, so the spatial
// coordinates are related to standard Bondi-Sachs coordinates by just the
// standard Cartesian to spherical Jacobian.
tuples::TaggedTuple<Tags::BondiBeta, Tags::BondiU, Tags::BondiW, Tags::BondiJ>
extract_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    double extraction_radius) noexcept;

// This function determines the time derivative of the Bondi-Sachs scalars
// from the time derivative of a Cartesian spacetime metric, the Cartesian
// metric, and Jacobian factors. This procedure assumes that the metric is
// already in null form, so the spatial coordinates are related to standard
// Bondi-Sachs coordinates by just the standard cartesian to spherical Jacobian.
tuples::TaggedTuple<::Tags::dt<Tags::BondiBeta>, ::Tags::dt<Tags::BondiU>,
                    ::Tags::dt<Tags::BondiW>, ::Tags::dt<Tags::BondiJ>>
extract_dt_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    double extraction_radius) noexcept;

// This function determines the radial derivative of the Bondi-Sachs scalars
// from the radial derivative of a Cartesian spacetime metric, the Cartesian
// metric, and Jacobian factors. This procedure assumes that the metric is
// already in null form, so the spatial coordinates are related to standard
// Bondi-Sachs coordinates by just the standard cartesian to spherical Jacobian.
tuples::TaggedTuple<Tags::Dr<Tags::BondiBeta>, Tags::Dr<Tags::BondiU>,
                    Tags::Dr<Tags::BondiW>, Tags::Dr<Tags::BondiJ>>
extract_dr_bondi_scalars_from_cartesian_metric(
    const tnsr::aa<DataVector, 3>& dr_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const CartesianiSphericalJ& inverse_jacobian,
    const CartesianiSphericalJ& dr_inverse_jacobian,
    double extraction_radius) noexcept;

// This function checks the consistency of the 3+1 ADM quantities in the
// `boundary_tuple` with quantities computed from `expected_spacetime_metric`,
// `expected_dt_spacetime_metric`, and `expected_d_spacetime_metric`. If the
// expected quantities are also extracted from the tuple, this simply checks
// that the boundary computation has produced a consistent set of quantities
// and has not generated NaNs or other pathological values (e.g. a degenerate
// spacetime metric) in the process.
template <typename... TupleTags>
void check_adm_metric_quantities(
    const tuples::TaggedTuple<TupleTags...>& boundary_tuple,
    const tnsr::aa<DataVector, 3>& expected_spacetime_metric,
    const tnsr::aa<DataVector, 3>& expected_dt_spacetime_metric,
    const tnsr::iaa<DataVector, 3>& expected_d_spacetime_metric) noexcept {
  // check the 3+1 quantities are computed correctly in the abstract base class
  // `WorldtubeData`
  const auto& dr_cartesian_coordinates =
      get<Tags::Dr<Tags::CauchyCartesianCoords>>(boundary_tuple);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(boundary_tuple);
  const auto& shift =
      get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(boundary_tuple);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          boundary_tuple);

  const auto expected_spatial_metric =
      gr::spatial_metric(expected_spacetime_metric);
  const auto expected_inverse_spatial_metric =
      determinant_and_inverse(expected_spatial_metric).second;
  const auto expected_shift =
      gr::shift(expected_spacetime_metric, expected_inverse_spatial_metric);
  const auto expected_lapse =
      gr::lapse(expected_shift, expected_spacetime_metric);
  CHECK_ITERABLE_APPROX(spatial_metric, expected_spatial_metric);
  CHECK_ITERABLE_APPROX(shift, expected_shift);
  CHECK_ITERABLE_APPROX(lapse, expected_lapse);

  const auto& pi =
      get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(boundary_tuple);
  const auto dt_spacetime_metric_from_pi =
      GeneralizedHarmonic::time_derivative_of_spacetime_metric(
          expected_lapse, expected_shift, pi, expected_d_spacetime_metric);
  CHECK_ITERABLE_APPROX(expected_dt_spacetime_metric,
                        dt_spacetime_metric_from_pi);
  // Check that the time derivative values are consistent with the Generalized
  // Harmonic `pi` -- these are redundant if the boundary calculation uses `pi`
  // to derive these, but it is not required to do so.
  const auto& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(boundary_tuple);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          boundary_tuple);
  const auto& dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      boundary_tuple);
  const auto expected_dt_spatial_metric =
      GeneralizedHarmonic::time_deriv_of_spatial_metric(
          expected_lapse, expected_shift, expected_d_spacetime_metric, pi);
  const auto expected_spacetime_unit_normal =
      gr::spacetime_normal_vector(expected_lapse, expected_shift);
  const auto expected_dt_lapse = GeneralizedHarmonic::time_deriv_of_lapse(
      expected_lapse, expected_shift, expected_spacetime_unit_normal,
      expected_d_spacetime_metric, pi);
  const auto expected_dt_shift = GeneralizedHarmonic::time_deriv_of_shift(
      expected_lapse, expected_shift, expected_inverse_spatial_metric,
      expected_spacetime_unit_normal, expected_d_spacetime_metric, pi);
  CHECK_ITERABLE_APPROX(dt_lapse, expected_dt_lapse);
  CHECK_ITERABLE_APPROX(dt_shift, expected_dt_shift);
  CHECK_ITERABLE_APPROX(dt_spatial_metric, expected_dt_spatial_metric);

  const auto& dr_lapse =
      get<Tags::Dr<gr::Tags::Lapse<DataVector>>>(boundary_tuple);
  const auto& dr_shift =
      get<Tags::Dr<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          boundary_tuple);
  const auto& dr_spatial_metric =
      get<Tags::Dr<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
          boundary_tuple);
  const auto expected_spatial_derivative_of_lapse =
      GeneralizedHarmonic::spatial_deriv_of_lapse(
          expected_lapse, expected_spacetime_unit_normal,
          expected_d_spacetime_metric);
  const auto expected_inverse_spacetime_metric = gr::inverse_spacetime_metric(
      expected_lapse, expected_shift, expected_inverse_spatial_metric);
  const auto expected_spatial_derivative_of_shift =
      GeneralizedHarmonic::spatial_deriv_of_shift(
          expected_lapse, expected_inverse_spacetime_metric,
          expected_spacetime_unit_normal, expected_d_spacetime_metric);
  DataVector expected_buffer =
      get<0>(dr_cartesian_coordinates) *
          get<0>(expected_spatial_derivative_of_lapse) +
      get<1>(dr_cartesian_coordinates) *
          get<1>(expected_spatial_derivative_of_lapse) +
      get<2>(dr_cartesian_coordinates) *
          get<2>(expected_spatial_derivative_of_lapse);
  CHECK_ITERABLE_APPROX(expected_buffer, get(dr_lapse));
  for (size_t i = 0; i < 3; ++i) {
    expected_buffer = get<0>(dr_cartesian_coordinates) *
                          expected_spatial_derivative_of_shift.get(0, i) +
                      get<1>(dr_cartesian_coordinates) *
                          expected_spatial_derivative_of_shift.get(1, i) +
                      get<2>(dr_cartesian_coordinates) *
                          expected_spatial_derivative_of_shift.get(2, i);
    CHECK_ITERABLE_APPROX(expected_buffer, dr_shift.get(i));
    for (size_t j = i; j < 3; ++j) {
      expected_buffer = get<0>(dr_cartesian_coordinates) *
                            expected_d_spacetime_metric.get(0, i + 1, j + 1) +
                        get<1>(dr_cartesian_coordinates) *
                            expected_d_spacetime_metric.get(1, i + 1, j + 1) +
                        get<2>(dr_cartesian_coordinates) *
                            expected_d_spacetime_metric.get(2, i + 1, j + 1);
      CHECK_ITERABLE_APPROX(expected_buffer, dr_spatial_metric.get(i, j));
    }
  }
}

}  // namespace TestHelpers
}  // namespace Solutions
}  // namespace Cce
