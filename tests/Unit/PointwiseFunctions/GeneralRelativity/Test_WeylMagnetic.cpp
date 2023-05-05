// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename DataType>
void make_random_tensors(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> grad_extrinsic_curvature,
    const gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
    const gsl::not_null<Scalar<DataType>*> sqrt_det_spatial_metric,
    const gsl::not_null<tnsr::II<DataType, 3>*> inverse_spatial_metric,
    const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-3.0, 3.0);
  const auto nn_distribution = make_not_null(&distribution);
  *grad_extrinsic_curvature = make_with_random_values<tnsr::ijj<DataType, 3>>(
      make_not_null(&generator), nn_distribution, used_for_size);

  std::uniform_real_distribution<> metric_distribution(-0.03, 0.03);
  const auto nn_metric_distribution = make_not_null(&metric_distribution);
  *spatial_metric = make_with_random_values<tnsr::ii<DataType, 3>>(
      make_not_null(&generator), nn_metric_distribution, used_for_size);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric->get(i, i) += 1.0;
  }

  std::uniform_real_distribution<> positive_distribution(0.1, 1.0);
  *sqrt_det_spatial_metric = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), positive_distribution, used_for_size);

  *inverse_spatial_metric = make_with_random_values<tnsr::II<DataType, 3>>(
      make_not_null(&generator), nn_metric_distribution, used_for_size);
}

template <typename DataType>
void test_compute_item_in_databox(const DataType& used_for_size) {
  TestHelpers::db::test_compute_tag<
      gr::Tags::WeylMagneticCompute<DataType, 3, Frame::Inertial>>(
      "WeylMagnetic");

  auto grad_extrinsic_curvature = make_with_value<tnsr::ijj<DataType, 3>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto spatial_metric = make_with_value<tnsr::ii<DataType, 3>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto sqrt_det_spatial_metric = make_with_value<Scalar<DataType>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto inverse_spatial_metric = make_with_value<tnsr::II<DataType, 3>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());

  make_random_tensors(make_not_null(&grad_extrinsic_curvature),
                      make_not_null(&spatial_metric),
                      make_not_null(&sqrt_det_spatial_metric),
                      make_not_null(&inverse_spatial_metric), used_for_size);

  const auto box = db::create<
      db::AddSimpleTags<::Tags::deriv<gr::Tags::ExtrinsicCurvature<DataType, 3>,
                                      tmpl::size_t<3>, Frame::Inertial>,
                        gr::Tags::SpatialMetric<DataType, 3>,
                        gr::Tags::SqrtDetSpatialMetric<DataType>,
                        gr::Tags::InverseSpatialMetric<DataType, 3>>,
      db::AddComputeTags<
          gr::Tags::WeylMagneticCompute<DataType, 3, Frame::Inertial>,
          gr::Tags::WeylMagneticScalarCompute<DataType, 3, Frame::Inertial>>>(
      grad_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric,
      inverse_spatial_metric);

  const auto expected_weyl_magnetic = gr::weyl_magnetic(
      grad_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric);
  CHECK_ITERABLE_APPROX(
      (db::get<gr::Tags::WeylMagnetic<DataType, 3, Frame::Inertial>>(box)),
      expected_weyl_magnetic);

  const auto expected_weyl_scalar =
      gr::weyl_magnetic_scalar(expected_weyl_magnetic, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX((db::get<gr::Tags::WeylMagneticScalar<DataType>>(box)),
                        expected_weyl_scalar);
}

template <size_t SpatialDim, typename DataType>
void test_weyl_magnetic(const DataType& used_for_size) {
  // Initialize input Tensor objects
  auto grad_extrinsic_curvature =
      make_with_value<tnsr::ijj<DataType, SpatialDim>>(
          used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto spatial_metric = make_with_value<tnsr::ii<DataType, SpatialDim>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto sqrt_det_spatial_metric = make_with_value<Scalar<DataType>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  auto inverse_spatial_metric = make_with_value<tnsr::II<DataType, 3>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());

  // Populating inputs with random values
  make_random_tensors(make_not_null(&grad_extrinsic_curvature),
                      make_not_null(&spatial_metric),
                      make_not_null(&sqrt_det_spatial_metric),
                      make_not_null(&inverse_spatial_metric), used_for_size);

  const auto cpp_weyl_magnetic = gr::weyl_magnetic(
      grad_extrinsic_curvature, spatial_metric, sqrt_det_spatial_metric);
  const auto scalar_weyl_magnetic =
      gr::weyl_magnetic_scalar(cpp_weyl_magnetic, inverse_spatial_metric);
  const auto python_weyl_magnetic =
      pypp::call<tnsr::ii<DataType, SpatialDim, Frame::Inertial>>(
          "WeylMagnetic", "weyl_magnetic_tensor", grad_extrinsic_curvature,
          spatial_metric, sqrt_det_spatial_metric);
  const auto python_weyl_scalar =
      pypp::call<Scalar<DataType>>("WeylMagneticScalar", "weyl_magnetic_scalar",
                                   cpp_weyl_magnetic, inverse_spatial_metric);
  CHECK_ITERABLE_APPROX(cpp_weyl_magnetic, python_weyl_magnetic);
  CHECK_ITERABLE_APPROX(scalar_weyl_magnetic, python_weyl_scalar);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.WeylMagnetic",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;

  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_weyl_magnetic, (3));
  test_compute_item_in_databox(d);
  test_compute_item_in_databox(dv);
}
