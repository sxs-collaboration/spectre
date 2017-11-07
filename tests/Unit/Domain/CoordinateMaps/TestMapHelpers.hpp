// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Helper functions for testing coordinate maps

#pragma once

#include <array>
#include <functional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map and a CoordinateMapBase, checks that the maps are equal by
 * downcasting `map_base` and then comparing to `map`. Returns false if the
 * downcast fails.
 */
template <typename Map>
bool are_maps_equal(
    const Map& map,
    const CoordinateMapBase<Frame::Logical, Frame::Inertial, Map::dim>&
        map_base) {
  const auto* map_derived = dynamic_cast<const Map*>(&map_base);
  return map_derived == nullptr ? false : (*map_derived == map);
}

/// \ingroup TestingFrameworkGroup
/// \brief Given two coordinate maps (but not their types), check that the maps
/// are equal by evaluating them at a random set of points.
template <typename SourceFrame, typename TargetFrame, size_t VolumeDim>
void check_if_maps_are_equal(
    const CoordinateMapBase<SourceFrame, TargetFrame, VolumeDim>& map_one,
    const CoordinateMapBase<SourceFrame, TargetFrame, VolumeDim>& map_two) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);

  for (size_t n = 0; n < 10; ++n) {
    tnsr::I<double, VolumeDim, SourceFrame> source_point{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      source_point.get(d) = real_dis(gen);
    }
    CAPTURE_PRECISE(source_point);
    CHECK_ITERABLE_APPROX(map_one(source_point), map_two(source_point));
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the jacobian gives expected results
 * when compared to the numerical derivative in each direction.
 */
template <typename Map>
void test_jacobian(const Map& map,
                   const std::array<double, Map::dim>& test_point) {
  // Our default approx value is too stringent for this test
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  const double dx = 1e-4;
  const auto jacobian = map.jacobian(test_point);
  for (size_t i = 0; i < Map::dim; ++i) {
    const auto numerical_deriv_i = numerical_derivative(map, test_point, i, dx);
    for (size_t j = 0; j < Map::dim; ++j) {
      INFO("i: " << i << " j: " << j);
      CHECK(jacobian.get(j, i) == local_approx(gsl::at(numerical_deriv_i, j)));
    }
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the inverse jacobian and jacobian
 * multiply together to produce the identity matrix
 */
template <typename Map>
void test_inv_jacobian(const Map& map,
                       const std::array<double, Map::dim>& test_point) {
  const auto jacobian = map.jacobian(test_point);
  const auto inv_jacobian = map.inv_jacobian(test_point);

  const auto expected_identity = [&jacobian, &inv_jacobian]() {
    std::array<std::array<double, Map::dim>, Map::dim> identity{};
    for (size_t i = 0; i < Map::dim; ++i) {
      for (size_t j = 0; j < Map::dim; ++j) {
        gsl::at(gsl::at(identity, i), j) = 0.;
        for (size_t k = 0; k < Map::dim; ++k) {
          gsl::at(gsl::at(identity, i), j) +=
              jacobian.get(i, k) * inv_jacobian.get(k, j);
        }
      }
    }
    return identity;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    for (size_t j = 0; j < Map::dim; ++j) {
      CHECK(gsl::at(gsl::at(expected_identity, i), j) ==
            approx(i == j ? 1. : 0.));
    }
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Checks that the CoordinateMap `map` functions as expected when used as
 * the template parameter to the `CoordinateMap` type.
 */
template <typename Map, typename... Args>
void test_coordinate_map_implementation(const Map& map) {
  const auto coord_map = make_coordinate_map<Frame::Logical, Frame::Grid>(map);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> real_dis(-1, 1);

  const auto test_point = [&gen, &real_dis] {
    std::array<double, Map::dim> p{};
    for (size_t i = 0; i < Map::dim; ++i) {
      gsl::at(p, i) = real_dis(gen);
    }
    return p;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    CAPTURE_PRECISE(gsl::at(test_point, i));
  }

  const auto test_point_tensor = [&test_point]() {
    tnsr::I<double, Map::dim, Frame::Logical> point_as_tensor{};
    for (size_t i = 0; i < Map::dim; ++i) {
      point_as_tensor.get(i) = gsl::at(test_point, i);
    }
    return point_as_tensor;
  }();

  for (size_t i = 0; i < Map::dim; ++i) {
    CHECK(coord_map(test_point_tensor).get(i) ==
          approx(gsl::at(map(test_point), i)));
    for (size_t j = 0; j < Map::dim; ++j) {
      CHECK(coord_map.jacobian(test_point_tensor).get(i, j) ==
            map.jacobian(test_point).get(i, j));
      CHECK(coord_map.inv_jacobian(test_point_tensor).get(i, j) ==
            map.inv_jacobian(test_point).get(i, j));
    }
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Checks that the CoordinateMap `map` functions as expected when used
 * with different argument types.
 */
template <bool CheckInverse = true, typename Map, typename... Args>
void test_coordinate_map_argument_types(
    const Map& map, const std::array<double, Map::dim>& test_point) {
  const auto make_array_data_vector = [](const auto& double_array) {
    std::array<DataVector, Map::dim> result;
    std::transform(double_array.begin(), double_array.end(), result.begin(),
                   [](const double x) { return DataVector{x, x}; });
    return result;
  };
  const auto make_tensor_data_vector = [](const auto& double_tensor) {
    using Arg = std::decay_t<decltype(double_tensor)>;
    Tensor<DataVector, typename Arg::symmetry, typename Arg::index_list> result;
    std::transform(double_tensor.begin(), double_tensor.end(), result.begin(),
                   [](const double x) { return DataVector{x, x}; });
    return result;
  };
  const auto add_reference_wrapper = [](const auto& unwrapped_array) {
    using Arg = std::decay_t<decltype(unwrapped_array)>;
    return make_array<std::reference_wrapper<const typename Arg::value_type>,
                      Map::dim>(unwrapped_array);
  };

  const auto mapped_point = map(test_point);
  CHECK_ITERABLE_APPROX(map(add_reference_wrapper(test_point)), mapped_point);
  CHECK_ITERABLE_APPROX(map(make_array_data_vector(test_point)),
                        make_array_data_vector(mapped_point));
  CHECK_ITERABLE_APPROX(
      map(add_reference_wrapper(make_array_data_vector(test_point))),
      make_array_data_vector(mapped_point));

  if (CheckInverse) {
    const auto expected = map.inverse(mapped_point);
    CHECK_ITERABLE_APPROX(map.inverse(add_reference_wrapper(mapped_point)),
                          expected);
    CHECK_ITERABLE_APPROX(map.inverse(make_array_data_vector(mapped_point)),
                          make_array_data_vector(expected));
    CHECK_ITERABLE_APPROX(map.inverse(add_reference_wrapper(
                              make_array_data_vector(mapped_point))),
                          make_array_data_vector(expected));
  }

  {
    const auto expected = map.jacobian(test_point);
    CHECK_ITERABLE_APPROX(map.jacobian(add_reference_wrapper(test_point)),
                          expected);
    CHECK_ITERABLE_APPROX(map.jacobian(make_array_data_vector(test_point)),
                          make_tensor_data_vector(expected));
    CHECK_ITERABLE_APPROX(
        map.jacobian(add_reference_wrapper(make_array_data_vector(test_point))),
        make_tensor_data_vector(expected));
  }

  {
    const auto expected = map.inv_jacobian(test_point);
    CHECK_ITERABLE_APPROX(map.inv_jacobian(add_reference_wrapper(test_point)),
                          expected);
    CHECK_ITERABLE_APPROX(map.inv_jacobian(make_array_data_vector(test_point)),
                          make_tensor_data_vector(expected));
    CHECK_ITERABLE_APPROX(map.inv_jacobian(add_reference_wrapper(
                              make_array_data_vector(test_point))),
                          make_tensor_data_vector(expected));
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Given a Map `map`, checks that the inverse map gives expected results
 */
template <typename Map, typename T>
void test_inverse_map(const Map& map,
                      const std::array<T, Map::dim>& test_point) {
  CHECK_ITERABLE_APPROX(test_point, map.inverse(map(test_point)));
}
