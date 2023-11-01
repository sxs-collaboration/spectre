// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t SpatialDim>
void test_lorentz_boost_matrix_random(const double& used_for_size) {
  tnsr::Ab<double, SpatialDim, Frame::NoFrame> (*f)(
      const tnsr::I<double, SpatialDim, Frame::NoFrame>&) =
      &sr::lorentz_boost_matrix<SpatialDim>;
  // The boost matrix is actually singular if the boost velocity is unity,
  // so let's ensure the speed never exceeds 0.999.
  constexpr double max_speed = 0.999;
  constexpr double upper = max_speed / static_cast<double>(SpatialDim);
  constexpr double lower = -upper;
  pypp::check_with_random_values<1>(f, "LorentzBoostMatrix",
                                    "lorentz_boost_matrix", {{{lower, upper}}},
                                    used_for_size);
}

template <size_t SpatialDim>
void test_lorentz_boost_matrix_analytic(const double& velocity_squared) {
  // Check that zero velocity returns an identity matrix
  const tnsr::I<double, SpatialDim, Frame::NoFrame> velocity_zero{0.0};
  const auto boost_matrix_zero = sr::lorentz_boost_matrix(velocity_zero);

  // Do not use DataStructures/Tensor/Identity.hpp, because identity returns a
  // spatial tensor, not a spacetime tensor
  tnsr::Ab<double, SpatialDim, Frame::NoFrame> identity_matrix{0.0};
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    identity_matrix.get(i, i) = 1.0;
  }
  CHECK_ITERABLE_APPROX(boost_matrix_zero, identity_matrix);

  // Check that the boost matrix inverse is the boost matrix with v->-v
  const tnsr::I<double, SpatialDim, Frame::NoFrame> velocity{
      sqrt(velocity_squared / static_cast<double>(SpatialDim))};
  const tnsr::I<double, SpatialDim, Frame::NoFrame> minus_velocity{
      -sqrt(velocity_squared / static_cast<double>(SpatialDim))};
  const auto boost_matrix = sr::lorentz_boost_matrix(velocity);
  const auto boost_matrix_minus = sr::lorentz_boost_matrix(minus_velocity);
  tnsr::Ab<double, SpatialDim, Frame::NoFrame> inverse_check{0.0};
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    for (size_t j = 0; j < SpatialDim + 1; ++j) {
      for (size_t k = 0; k < SpatialDim + 1; ++k) {
        inverse_check.get(i, j) +=
            boost_matrix.get(i, k) * boost_matrix_minus.get(k, j);
      }
    }
  }
  CHECK_ITERABLE_APPROX(inverse_check, identity_matrix);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void test_lorentz_boost(const std::array<double, SpatialDim> velocity,
                        const std::array<double, SpatialDim> velocity_2) {
  const DataVector used_for_size{3., 4., 5.};

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-0.2, 0.2);

  auto covariant_vector =
      make_with_random_values<tnsr::a<DataType, SpatialDim, Frame>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);

  tnsr::a<DataType, SpatialDim, Frame> boosted_covariant_vector;
  sr::lorentz_boost<DataType, SpatialDim, Frame>(
      make_not_null(&boosted_covariant_vector), covariant_vector, velocity);

  // Check that applying the inverse boost returns the original vector
  tnsr::a<DataType, SpatialDim, Frame> unboosted_covariant_vector;
  sr::lorentz_boost<DataType, SpatialDim, Frame>(
      make_not_null(&unboosted_covariant_vector), boosted_covariant_vector,
      -velocity);
  CHECK_ITERABLE_APPROX(unboosted_covariant_vector, covariant_vector);

  // Check boost of the spatial vector
  tnsr::I<DataType, SpatialDim, Frame> spatial_vector =
      make_with_value<tnsr::I<DataType, SpatialDim, Frame>>(used_for_size, 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    spatial_vector.get(i) = covariant_vector.get(i + 1);
  }
  // Lower index, i.e. v_0
  const double vector_component_0 = get<0>(covariant_vector);

  tnsr::I<DataType, SpatialDim, Frame> boosted_spatial_vector =
      make_with_value<tnsr::I<DataType, SpatialDim, Frame>>(used_for_size, 0.0);

  sr::lorentz_boost<DataType, SpatialDim, Frame>(
      make_not_null(&boosted_spatial_vector), spatial_vector,
      vector_component_0, velocity);

  tnsr::I<DataType, SpatialDim, Frame> expected_spatial_vector =
      make_with_value<tnsr::I<DataType, SpatialDim, Frame>>(used_for_size, 0.0);
  for (size_t i = 0; i < SpatialDim; ++i) {
    expected_spatial_vector.get(i) = boosted_covariant_vector.get(i + 1);
  }
  CHECK_ITERABLE_APPROX(expected_spatial_vector, boosted_spatial_vector);

  // Check boost on rank-2 matrix by constructing one with an outer tensor
  // product, i.e. T_{ab} = v_{a} u_{b}
  tnsr::ab<DataType, SpatialDim, Frame> tensor =
      make_with_value<tnsr::ab<DataType, SpatialDim, Frame>>(used_for_size,
                                                             0.0);
  tnsr::ab<DataType, SpatialDim, Frame> boosted_tensor =
      make_with_value<tnsr::ab<DataType, SpatialDim, Frame>>(used_for_size,
                                                             0.0);
  tnsr::ab<DataType, SpatialDim, Frame> expected_tensor =
      make_with_value<tnsr::ab<DataType, SpatialDim, Frame>>(used_for_size,
                                                             0.0);
  // We need a second covariant vector
  auto covariant_vector_2 =
      make_with_random_values<tnsr::a<DataType, SpatialDim, Frame>>(
          make_not_null(&generator), make_not_null(&distribution),
          used_for_size);
  // We boost it as well, but with possibly another velocity
  tnsr::a<DataType, SpatialDim, Frame> boosted_covariant_vector_2;
  sr::lorentz_boost<DataType, SpatialDim, Frame>(
      make_not_null(&boosted_covariant_vector_2), covariant_vector_2,
      velocity_2);

  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    for (size_t j = 0; j < SpatialDim + 1; ++j) {
      tensor.get(i, j) = covariant_vector.get(i) * covariant_vector_2.get(j);
      expected_tensor.get(i, j) =
          boosted_covariant_vector.get(i) * boosted_covariant_vector_2.get(j);
    }
  }

  sr::lorentz_boost<DataType, SpatialDim, Frame>(make_not_null(&boosted_tensor),
                                                 tensor, velocity, velocity_2);
  CHECK_ITERABLE_APPROX(expected_tensor, boosted_tensor);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.SpecialRelativity.LorentzBoostMatrix",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/SpecialRelativity/");

  // CHECK_FOR_DOUBLES passes a double named d to the test function.
  // For test_lorentz_boost_matrix_random(), this double is only used for
  // size, so here set it to a signaling NaN.
  double d(std::numeric_limits<double>::signaling_NaN());
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_random, (1, 2, 3));

  const double small_velocity_squared = 5.0e-6;
  const double large_velocity_squared = 0.99;

  // The analytic test function uses the double that CHECK_FOR_DOUBLES passes to
  // set the magnitude of the velocity. Here, we test both for a small and for
  // a large velocity.
  d = small_velocity_squared;
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_analytic, (1, 2, 3));

  d = large_velocity_squared;
  CHECK_FOR_DOUBLES(test_lorentz_boost_matrix_analytic, (1, 2, 3));
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.SpecialRelativity.LorentzBoost",
                  "[PointwiseFunctions][Unit]") {
  const std::array<double, 3> velocity{{0.1, -0.4, 0.3}};
  const std::array<double, 3> velocity_2{{0.2, -0.1, -0.5}};
  test_lorentz_boost<double, 3, Frame::Inertial>(velocity, velocity_2);
}
