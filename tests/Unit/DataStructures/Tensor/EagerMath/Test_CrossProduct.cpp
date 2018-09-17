// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/CrossProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor

template <typename DataType>
void check_cross_product(const DataType& used_for_size) noexcept {
  // Make constants used to create vectors and one_forms
  const auto zero = make_with_value<DataType>(used_for_size, 0.0);
  const auto one_over_three_hundred_twenty_seven =
      make_with_value<DataType>(used_for_size, 1.0 / 327.0);
  const auto one_half = make_with_value<DataType>(used_for_size, 0.5);
  const auto minus_one_half = make_with_value<DataType>(used_for_size, -0.5);
  const auto one = make_with_value<DataType>(used_for_size, 1.0);
  const auto minus_one = make_with_value<DataType>(used_for_size, -1.0);
  const auto two = make_with_value<DataType>(used_for_size, 2.0);
  const auto minus_two = make_with_value<DataType>(used_for_size, -2.0);
  const auto minus_three = make_with_value<DataType>(used_for_size, -3.0);
  const auto four = make_with_value<DataType>(used_for_size, 4.0);
  const auto minus_five = make_with_value<DataType>(used_for_size, -5.0);
  const auto twelve = make_with_value<DataType>(used_for_size, 12.0);
  const auto twenty_two = make_with_value<DataType>(used_for_size, 22.0);
  const auto minus_thirty_three =
      make_with_value<DataType>(used_for_size, -33.0);
  const auto forty_four = make_with_value<DataType>(used_for_size, 44.0);
  const auto sixty_four = make_with_value<DataType>(used_for_size, 64.0);

  // Test Euclidean cross product for a known generic case
  const tnsr::I<DataType, 3, Frame::Grid> vector_a{
      {{minus_three, twelve, four}}};
  const tnsr::I<DataType, 3, Frame::Grid> vector_b{{{four, minus_five, two}}};
  const tnsr::i<DataType, 3, Frame::Grid> covector_b{{{four, minus_five, two}}};
  const tnsr::I<DataType, 3, Frame::Grid> vector_expected{
      {{forty_four, twenty_two, minus_thirty_three}}};
  const tnsr::i<DataType, 3, Frame::Grid> covector_expected{
      {{forty_four, twenty_two, minus_thirty_three}}};
  CHECK_ITERABLE_APPROX(cross_product(vector_a, vector_b), vector_expected);
  CHECK_ITERABLE_APPROX(cross_product(vector_a, covector_b), covector_expected);

  // Test curved-space cross product for a known generic case
  const tnsr::II<DataType, 3, Frame::Grid> inverse_metric = [&used_for_size]() {
    auto tensor =
        make_with_value<tnsr::II<DataType, 3, Frame::Grid>>(used_for_size, 0.0);
    get<0, 0>(tensor) = 2.0;
    get<0, 1>(tensor) = -3.0;
    get<0, 2>(tensor) = 4.0;
    get<1, 1>(tensor) = -5.0;
    get<1, 2>(tensor) = -12.0;
    get<2, 2>(tensor) = -13.0;
    return tensor;
  }();

  const auto det_metric = Scalar<DataType>{one_over_three_hundred_twenty_seven};

  // The known case for curved-space happens to involve the same vector_a
  // but different vector_b as for the Euclidean cross product test
  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_b{
      {{minus_five, four, two}}};
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_b =
      [&used_for_size]() {
        auto tensor = make_with_value<tnsr::i<DataType, 3, Frame::Grid>>(
            used_for_size, 0.0);
        get<0>(tensor) = 53.0 / 109.0;
        get<1>(tensor) = 97.0 / 109.0;
        get<2>(tensor) = -90.0 / 109.0;
        return tensor;
      }();

  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_expected =
      [&det_metric]() {
        auto tensor = make_with_value<tnsr::I<DataType, 3, Frame::Grid>>(
            get(det_metric), 0.0);
        get<0>(tensor) = 250.0 * sqrt(get(det_metric));
        get<1>(tensor) = -530.0 * sqrt(get(det_metric));
        get<2>(tensor) = -424.0 * sqrt(get(det_metric));
        return tensor;
      }();
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_expected =
      [&det_metric]() {
        auto tensor = make_with_value<tnsr::i<DataType, 3, Frame::Grid>>(
            get(det_metric), 0.0);
        get<0>(tensor) = 8.0 * sqrt(get(det_metric));
        get<1>(tensor) = -14.0 * sqrt(get(det_metric));
        get<2>(tensor) = 48.0 * sqrt(get(det_metric));
        return tensor;
      }();

  CHECK_ITERABLE_APPROX(
      cross_product(vector_a, curved_vector_b, inverse_metric, det_metric),
      curved_vector_expected);
  CHECK_ITERABLE_APPROX(
      cross_product(vector_a, curved_covector_b, inverse_metric, det_metric),
      curved_covector_expected);

  // Test Euclidean cross product for orthogonal unit vectors
  const tnsr::I<DataType, 3, Frame::Grid> vector_x_hat{{{one, zero, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> covector_x_hat{{{one, zero, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> vector_y_hat{{{zero, one, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> covector_y_hat{{{zero, one, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> vector_z_hat{{{zero, zero, one}}};
  const tnsr::I<DataType, 3, Frame::Grid> covector_z_hat{{{zero, zero, one}}};
  const tnsr::I<DataType, 3, Frame::Grid> vector_minus_z_hat{
      {{zero, zero, minus_one}}};
  const tnsr::I<DataType, 3, Frame::Grid> covector_minus_z_hat{
      {{zero, zero, minus_one}}};
  CHECK_ITERABLE_APPROX(cross_product(vector_x_hat, vector_y_hat),
                        vector_z_hat);
  CHECK_ITERABLE_APPROX(cross_product(vector_x_hat, covector_y_hat),
                        covector_z_hat);
  CHECK_ITERABLE_APPROX(cross_product(vector_y_hat, vector_x_hat),
                        vector_minus_z_hat);
  CHECK_ITERABLE_APPROX(cross_product(vector_y_hat, covector_x_hat),
                        covector_minus_z_hat);

  // Test curved-space cross product for orthogonal unit vectors
  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_x_hat{
      {{one_half, zero, zero}}};
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_x_hat{
      {{two, zero, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_y_hat{
      {{zero, one_half, zero}}};
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_y_hat{
      {{zero, two, zero}}};
  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_z_hat{
      {{zero, zero, one_half}}};
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_z_hat{
      {{zero, zero, two}}};
  const tnsr::I<DataType, 3, Frame::Grid> curved_vector_minus_z_hat{
      {{zero, zero, minus_one_half}}};
  const tnsr::i<DataType, 3, Frame::Grid> curved_covector_minus_z_hat{
      {{zero, zero, minus_two}}};

  const tnsr::II<DataType, 3, Frame::Grid> inverse_metric_simple =
      [&used_for_size]() {
        auto tensor = make_with_value<tnsr::II<DataType, 3, Frame::Grid>>(
            used_for_size, 0.0);
        get<0, 0>(tensor) = 0.25;
        get<1, 1>(tensor) = 0.25;
        get<2, 2>(tensor) = 0.25;
        return tensor;
      }();
  const auto det_metric_simple = Scalar<DataType>{sixty_four};

  CHECK_ITERABLE_APPROX(cross_product(curved_vector_x_hat, curved_vector_y_hat,
                                      inverse_metric_simple, det_metric_simple),
                        curved_vector_z_hat);
  CHECK_ITERABLE_APPROX(
      cross_product(curved_vector_x_hat, curved_covector_y_hat,
                    inverse_metric_simple, det_metric_simple),
      curved_covector_z_hat);
  CHECK_ITERABLE_APPROX(cross_product(curved_vector_y_hat, curved_vector_x_hat,
                                      inverse_metric_simple, det_metric_simple),
                        curved_vector_minus_z_hat);
  CHECK_ITERABLE_APPROX(
      cross_product(curved_vector_y_hat, curved_covector_x_hat,
                    inverse_metric_simple, det_metric_simple),
      curved_covector_minus_z_hat);

  // Test c++ vs. python using random values
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataType, 3, Frame::Grid> (*)(
          const tnsr::I<DataType, 3, Frame::Grid>&,
          const tnsr::I<DataType, 3, Frame::Grid>&)>(
          &cross_product<DataType, SpatialIndex<3, UpLo::Up, Frame::Grid>>),
      "numpy", "cross", {{{-10.0, 10.0}}}, vector_x_hat);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, 3, Frame::Grid> (*)(
          const tnsr::I<DataType, 3, Frame::Grid>&,
          const tnsr::i<DataType, 3, Frame::Grid>&)>(
          &cross_product<DataType, SpatialIndex<3, UpLo::Up, Frame::Grid>>),
      "numpy", "cross", {{{-10.0, 10.0}}}, vector_x_hat);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<DataType, 3, Frame::Grid> (*)(
          const tnsr::I<DataType, 3, Frame::Grid>&,
          const tnsr::I<DataType, 3, Frame::Grid>&,
          const tnsr::II<DataType, 3, Frame::Grid>&, const Scalar<DataType>&)>(
          &cross_product<DataType, SpatialIndex<3, UpLo::Up, Frame::Grid>>),
      "TestFunctions", "cross_product_up", {{{1.0, 10.0}}},
      curved_vector_x_hat);
  pypp::check_with_random_values<1>(
      static_cast<tnsr::i<DataType, 3, Frame::Grid> (*)(
          const tnsr::I<DataType, 3, Frame::Grid>&,
          const tnsr::i<DataType, 3, Frame::Grid>&,
          const tnsr::II<DataType, 3, Frame::Grid>&, const Scalar<DataType>&)>(
          &cross_product<DataType, SpatialIndex<3, UpLo::Up, Frame::Grid>>),
      "TestFunctions", "cross_product_lo", {{{1.0, 10.0}}},
      curved_vector_x_hat);
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.CrossProduct",
                  "[DataStructures][Unit]") {
  // Set up a python environment for check_with_random_values
  pypp::SetupLocalPythonEnvironment local_python_env(
      "DataStructures/Tensor/EagerMath/");
  check_cross_product(std::numeric_limits<double>::signaling_NaN());
  check_cross_product(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
