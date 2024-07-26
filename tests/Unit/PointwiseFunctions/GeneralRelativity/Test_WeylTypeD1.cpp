// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylTypeD1.hpp"

namespace {
template <size_t SpatialDim, typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) {
  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylTypeD1Compute<DataType, SpatialDim, Frame::Inertial>>(
      "WeylTypeD1");
  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylTypeD1ScalarCompute<DataType, SpatialDim, Frame::Inertial>>(
      "WeylTypeD1Scalar");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto weyl_electric =
      make_with_random_values<tnsr::ii<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto spatial_metric =
      make_with_random_values<tnsr::ii<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);
  const auto inverse_spatial_metric =
      make_with_random_values<tnsr::II<DataType, SpatialDim>>(
          nn_generator, nn_distribution, used_for_size);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::WeylElectric<DataType, SpatialDim, Frame::Inertial>,
          gr::Tags::SpatialMetric<DataType, SpatialDim, Frame::Inertial>,
          gr::Tags::InverseSpatialMetric<DataType, SpatialDim,
                                         Frame::Inertial>>,
      db::AddComputeTags<
          gr::Tags::WeylTypeD1Compute<DataType, SpatialDim, Frame::Inertial>,
          gr::Tags::WeylTypeD1ScalarCompute<DataType, SpatialDim,
                                            Frame::Inertial>>>(
      weyl_electric, spatial_metric, inverse_spatial_metric);

  const auto expected =
      gr::weyl_type_D1(weyl_electric, spatial_metric, inverse_spatial_metric);
  const auto expected_scalar =
      gr::weyl_type_D1_scalar(expected, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::WeylTypeD1<DataType, SpatialDim, Frame::Inertial>>(
          box)),
      expected);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::WeylTypeD1Scalar<DataType>>(box)),
                        expected_scalar);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_type_D1(const DataType& used_for_size) {
  tnsr::ii<DataType, SpatialDim, Frame::Inertial> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_type_D1<DataType, SpatialDim, Frame::Inertial>;
  pypp::check_with_random_values<1>(f, "WeylTypeD1", "weyl_type_D1",
                                    {{{-1., 1.}}}, used_for_size);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_type_D1_scalar(const DataType& used_for_size) {
  Scalar<DataType> (*f)(
      const tnsr::ii<DataType, SpatialDim, Frame::Inertial>&,
      const tnsr::II<DataType, SpatialDim, Frame::Inertial>&) =
      &gr::weyl_type_D1_scalar<DataType, SpatialDim, Frame::Inertial>;
  pypp::check_with_random_values<1>(f, "WeylTypeD1Scalar",
                                    "weyl_type_D1_scalar", {{{-1., 1.}}},
                                    used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylTypeD1",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_type_D1, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_type_D1_scalar, (1, 2, 3));
  test_compute_item_in_databox<3>(d);
  test_compute_item_in_databox<3>(dv);
}
