// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace {
template <size_t Dim, typename DataType>
void test_compute_spacetime_metric(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<tnsr::aa<DataType, Dim, Frame::Inertial> (*)(
          const Scalar<DataType>&,
          const tnsr::I<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&)>(
          &gr::spacetime_metric<Dim, Frame::Inertial, DataType>),
      "ComputeSpacetimeQuantities", "spacetime_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_inverse_spacetime_metric(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &gr::inverse_spacetime_metric<Dim, Frame::Inertial, DataType>,
      "ComputeSpacetimeQuantities", "inverse_spacetime_metric", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_derivatives_of_spacetime_metric(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &gr::derivatives_of_spacetime_metric<Dim, Frame::Inertial, DataType>,
      "ComputeSpacetimeQuantities", "derivatives_of_spacetime_metric",
      {{{-10., 10.}}}, used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_spacetime_normal_vector(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &gr::spacetime_normal_vector<Dim, Frame::Inertial, DataType>,
      "ComputeSpacetimeQuantities", "spacetime_normal_vector", {{{-10., 10.}}},
      used_for_size);
}
template <size_t Dim, typename DataType>
void test_compute_spacetime_normal_one_form(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto lapse = make_with_random_values<Scalar<DataType>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);

  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<Dim, Frame::Inertial>(lapse);
  CHECK_ITERABLE_APPROX(spacetime_normal_one_form.get(0), -lapse.get());
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(spacetime_normal_one_form.get(i + 1),
                          make_with_value<DataType>(used_for_size, 0.));
  }
}
template <size_t Dim, typename DataType>
void test_compute_extrinsic_curvature(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &gr::extrinsic_curvature<Dim, Frame::Inertial, DataType>,
      "ComputeSpacetimeQuantities", "extrinsic_curvature", {{{-10., 10.}}},
      used_for_size);
}

template <size_t Dim, typename T>
void test_compute_spatial_metric_lapse_shift(const T& used_for_size) {
  // Set up random values for lapse, shift, and spatial_metric.
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  std::uniform_real_distribution<> dist_positive(1., 2.);
  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);
  const auto nn_dist_positive = make_not_null(&dist_positive);

  const auto lapse = make_with_random_values<Scalar<T>>(
      nn_generator, nn_dist_positive, used_for_size);
  const auto shift = make_with_random_values<tnsr::I<T, Dim>>(
      nn_generator, nn_dist, used_for_size);
  const auto spatial_metric = [&]() {
    auto spatial_metric_l = make_with_random_values<tnsr::ii<T, Dim>>(
        nn_generator, nn_dist, used_for_size);
    // Make sure spatial_metric isn't singular by adding
    // large enough positive diagonal values.
    for (size_t i = 0; i < Dim; ++i) {
      spatial_metric_l.get(i, i) += 4.0;
    }
    return spatial_metric_l;
  }();

  // Make spacetime metric from spatial metric, lapse, and shift.
  // Then go backwards and compute the spatial metric, lapse, and shift
  // and make sure we get back the original values.
  const auto psi = gr::spacetime_metric(lapse, shift, spatial_metric);

  // Here are the functions we are testing.
  const auto spatial_metric_test = gr::spatial_metric(psi);
  const auto shift_test =
      gr::shift(psi, determinant_and_inverse(spatial_metric).second);
  const auto lapse_test = gr::lapse(shift, psi);

  CHECK_ITERABLE_APPROX(spatial_metric, spatial_metric_test);
  CHECK_ITERABLE_APPROX(shift, shift_test);
  CHECK_ITERABLE_APPROX(lapse, lapse_test);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.SpacetimeDecomp",
                  "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_inverse_spacetime_metric,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_derivatives_of_spacetime_metric, (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_normal_vector,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spacetime_normal_one_form,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_spatial_metric_lapse_shift,
                                    (1, 2, 3));
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_compute_extrinsic_curvature,
                                    (1, 2, 3));

  // Check that compute items work correctly in the DataBox
  // First, check that the names are correct
  CHECK(gr::Tags::SpacetimeMetricCompute<3, Frame::Inertial,
                                         DataVector>::name() ==
        "SpacetimeMetric");
  CHECK(
      gr::Tags::SpatialMetricCompute<3, Frame::Inertial, DataVector>::name() ==
      "SpatialMetric");
  CHECK(gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial,
                                         DataVector>::name() ==
        "InverseSpacetimeMetric");
  CHECK(
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>::name() ==
      "InverseSpatialMetric");
  CHECK(gr::Tags::ShiftCompute<3, Frame::Inertial, DataVector>::name() ==
        "Shift");
  CHECK(gr::Tags::LapseCompute<3, Frame::Inertial, DataVector>::name() ==
        "Lapse");
  CHECK(gr::Tags::SpacetimeNormalOneFormCompute<3, Frame::Inertial,
                                                DataVector>::name() ==
        "SpacetimeNormalOneForm");
  CHECK(
      gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>::name() ==
      "SpacetimeNormalVector");

  // Second, put the compute items into a data box and check that they
  // put the correct results
  // Let's start with a known spacetime metric, and then test the
  // compute items that depend on the spacetime metric
  DataVector test_vector{5.0, 4.0};
  auto spacetime_metric =
      make_with_value<tnsr::aa<DataVector, 3, Frame::Inertial>>(test_vector,
                                                                0.0);
  get<0, 0>(spacetime_metric) = -1.5;
  get<0, 1>(spacetime_metric) = 0.1;
  get<0, 2>(spacetime_metric) = 0.2;
  get<0, 3>(spacetime_metric) = 0.3;
  get<1, 1>(spacetime_metric) = 1.4;
  get<1, 2>(spacetime_metric) = 0.2;
  get<1, 3>(spacetime_metric) = 0.1;
  get<2, 2>(spacetime_metric) = 1.3;
  get<2, 3>(spacetime_metric) = 0.1;
  get<3, 3>(spacetime_metric) = 1.2;

  const auto& spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto& det_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& sqrt_det_spatial_metric =
      Scalar<DataVector>{sqrt(get(det_and_inverse_spatial_metric.first))};
  const auto& inverse_spatial_metric = det_and_inverse_spatial_metric.second;
  const auto& shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto& lapse = gr::lapse(shift, spacetime_metric);
  const auto& inverse_spacetime_metric =
      gr::inverse_spacetime_metric(lapse, shift, inverse_spatial_metric);

  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>,
      db::AddComputeTags<
          gr::Tags::SpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::DetAndInverseSpatialMetricCompute<3, Frame::Inertial,
                                                      DataVector>,
          gr::Tags::SqrtDetSpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpatialMetricCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::ShiftCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::LapseCompute<3, Frame::Inertial, DataVector>,
          gr::Tags::InverseSpacetimeMetricCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalOneFormCompute<3, Frame::Inertial,
                                                  DataVector>,
          gr::Tags::SpacetimeNormalVectorCompute<3, Frame::Inertial,
                                                 DataVector>>>(
      spacetime_metric);
  CHECK(db::get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(box) ==
        spatial_metric);
  CHECK(
      db::get<
          gr::Tags::DetAndInverseSpatialMetric<3, Frame::Inertial, DataVector>>(
          box) == det_and_inverse_spatial_metric);
  CHECK(db::get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(box) ==
        sqrt_det_spatial_metric);
  CHECK(db::get<gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>>(
            box) == inverse_spatial_metric);
  CHECK(db::get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(box) == shift);
  CHECK(db::get<gr::Tags::Lapse<DataVector>>(box) == lapse);
  CHECK(
      db::get<gr::Tags::InverseSpacetimeMetric<3, Frame::Inertial, DataVector>>(
          box) == inverse_spacetime_metric);
  CHECK(
      db::get<gr::Tags::SpacetimeNormalOneForm<3, Frame::Inertial, DataVector>>(
          box) ==
      gr::spacetime_normal_one_form<3, Frame::Inertial, DataVector>(lapse));
  CHECK(
      db::get<gr::Tags::SpacetimeNormalVector<3, Frame::Inertial, DataVector>>(
          box) ==
      gr::spacetime_normal_vector<3, Frame::Inertial, DataVector>(lapse,
                                                                  shift));

  // Now let's put the lapse, shift, and spatial metric into the databox
  // and test that we can compute the correct spacetime metric
  const auto second_box = db::create<
      db::AddSimpleTags<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                        gr::Tags::Lapse<DataVector>,
                        gr::Tags::Shift<3, Frame::Inertial, DataVector>>,
      db::AddComputeTags<
          gr::Tags::SpacetimeMetricCompute<3, Frame::Inertial, DataVector>>>(
      spatial_metric, lapse, shift);
  CHECK(db::get<gr::Tags::SpacetimeMetric<3, Frame::Inertial, DataVector>>(
            second_box) == spacetime_metric);
}
