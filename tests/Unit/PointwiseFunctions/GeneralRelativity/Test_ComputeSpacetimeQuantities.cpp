// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

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
}
