// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"

// IWYU pragma: no_include <boost/preprocessor/arithmetic/dec.hpp>
// IWYU pragma: no_include <boost/preprocessor/repetition/enum.hpp>
// IWYU pragma: no_include <boost/preprocessor/tuple/reverse.hpp>

namespace {
template <size_t SpatialDim, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) {
  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylElectricCompute<DataType, SpatialDim, Frame::Inertial>>(
      "WeylElectric");
  TestHelpers::db::test_compute_tag<gr::Tags::WeylElectricScalarCompute<
      DataType, SpatialDim, Frame::Inertial>>("WeylElectricScalar");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto spatial_ricci =
      make_with_random_values<tnsr::ii<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto extrinsic_curvature =
      make_with_random_values<tnsr::ii<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto inv_spatial_metric =
      make_with_random_values<tnsr::II<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);

  const auto box = db::create<
      db::AddSimpleTags<gr::Tags::SpatialRicci<DataType, SpatialDim>,
                        gr::Tags::ExtrinsicCurvature<DataType, SpatialDim>,
                        gr::Tags::InverseSpatialMetric<DataType, SpatialDim>>,
      db::AddComputeTags<
          gr::Tags::WeylElectricCompute<DataType, SpatialDim, Frame::Inertial>,
          gr::Tags::WeylElectricScalarCompute<DataType, SpatialDim,
                                              Frame::Inertial>>>(
      spatial_ricci, extrinsic_curvature, inv_spatial_metric);

  const auto expected =
      gr::weyl_electric(spatial_ricci, extrinsic_curvature, inv_spatial_metric);
  const auto expected_scalar =
      gr::weyl_electric_scalar(expected, inv_spatial_metric);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::WeylElectric<DataType, SpatialDim>>(box)), expected);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::WeylElectricScalar<DataType>>(box)),
                        expected_scalar);
}
template <size_t SpatialDim, typename DataType>
void test_weyl_electric(const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_electric<DataType, SpatialDim, Frame::Inertial>;
  pypp::check_with_random_values<1>(f, "WeylElectric", "weyl_electric_tensor",
                                    {{{-1., 1.}}}, used_for_size);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_electric_scalar(const DataType& used_for_size) {
  Scalar<DataType> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_electric_scalar<DataType, SpatialDim, Frame::Inertial>;
  pypp::check_with_random_values<1>(f, "WeylElectricScalar",
                                    "weyl_electric_scalar", {{{-1., 1.}}},
                                    used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylElectric",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_electric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_electric_scalar, (1, 2, 3));
  test_compute_item_in_databox<3>(d);
  test_compute_item_in_databox<3>(dv);
}
