// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"

namespace {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_propagating_plus_wrapper(
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij) {
  return gr::weyl_propagating<SpatialDim, Frame, DataType>(
      ricci, extrinsic_curvature, inverse_spatial_metric,
      cov_deriv_extrinsic_curvature, unit_interface_normal_vector,
      projection_IJ, projection_ij, projection_Ij, 1.);
}
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ii<DataType, SpatialDim, Frame> weyl_propagating_minus_wrapper(
    const tnsr::ii<DataType, SpatialDim, Frame>& ricci,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijj<DataType, SpatialDim, Frame>& cov_deriv_extrinsic_curvature,
    const tnsr::I<DataType, SpatialDim, Frame>& unit_interface_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& projection_IJ,
    const tnsr::ii<DataType, SpatialDim, Frame>& projection_ij,
    const tnsr::Ij<DataType, SpatialDim, Frame>& projection_Ij) {
  return gr::weyl_propagating<SpatialDim, Frame, DataType>(
      ricci, extrinsic_curvature, inverse_spatial_metric,
      cov_deriv_extrinsic_curvature, unit_interface_normal_vector,
      projection_IJ, projection_ij, projection_Ij, -1.);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_propagating(const DataType& used_for_size) {
  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ijj<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::Ij<DataType, SpatialDim, Frame::Inertial>&) =
        &weyl_propagating_plus_wrapper<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "GeneralRelativity.WeylPropagating",
                                      "weyl_propagating_mode_plus",
                                      {{{-1., 1.}}}, used_for_size);
  }
  {
    tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ijj<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::I<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::II<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
        const tnsr::Ij<DataType, SpatialDim, Frame::Inertial>&) =
        &weyl_propagating_minus_wrapper<SpatialDim, Frame::Inertial, DataType>;
    pypp::check_with_random_values<1>(f, "GeneralRelativity.WeylPropagating",
                                      "weyl_propagating_mode_minus",
                                      {{{-1., 1.}}}, used_for_size);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylPropagating",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env("PointwiseFunctions/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_propagating, (1, 2, 3));
}
