// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <optional>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#pragma once

/// \ref protocols related to coordinate maps
namespace domain::CoordinateMaps::protocols {

/*!
 * \ingroup ProtocolsGroup
 * \brief Defines the interface for transition functions used by the shape map.
 *
 * \details This protocol defines the interface that a class must conform to so
 * that it may be used as a transition function by the shape map. Different
 * domains require the shape map to fall off towards the boundary in different
 * ways. This behavior is controlled by the transition function. It is also
 * needed to find the inverse of the shape map. Since the shape map preserves
 * angles, the problem of finding its inverse reduces to the 1-dimensional
 * problem of finding the original radius from the mapped radius. The mapped
 * radius \f$\tilde{r}\f$ is related to the original \f$r\f$ radius by:
 * \f{equation}{
 * \tilde{r} = r (1 - f(r,\theta,\phi) \sum_{lm} \lambda_{lm}(t)Y_{lm}(\theta,
 * \phi))
 * \f}
 * where everything but \f$\tilde{r}\f$ and \f$r\f$ can be treated as constant.
 * This equation is inverted in `original_radius_over_radius` and also divided
 * by the mapped radius to simplify calculations in the shape map.
 *
 * The transition function must also be able to compute its gradient and the
 * value of the function divided by the radius. Care must be taken that this
 * does not divide by zero.
 *
 * All member functions are templated on a parameter T which can be both a
 * `double` or a `DataVector`. The `ConformingType` needs to have these member
 * functions:
 * - `T operator()(std::array<T> source_coordinates)`: Evaluate the transition
 * function at the Cartesian coordinates.
 * - `T original_radius_over_radius(std::array<T> target_coordinates)`:
 * Evaluates the original radius from the mapped radius by inverting the shape
 * map. It also divides by the mapped radius to simplify calculations in the
 * shape map.
 * - `std::array<T> gradient(std::array<T> source_coordinates)`: Evaluate the
 * gradient of the transition function at the Cartesian coordinates.
 * - `T map_over_radius(std::array<T> source_coordinates)`: Evaluate the
 * transition function at the Cartesian coordinates divided by the radius. Care
 * must be taken not to divide by zero.
 * - `bool operator==(const ConformingType& lhs, const ConformingType& rhs)`
 */
struct TransitionFunc {
  template <typename ConformingType>
  struct test {
    using call_return_type_double = decltype(std::declval<ConformingType>()(
        std::declval<std::array<double, 3>>()));
    using call_return_type_DataVector = decltype(std::declval<ConformingType>()(
        std::declval<std::array<DataVector, 3>>()));

    using original_radius_over_radius_return_type =
        decltype(std::declval<ConformingType>().original_radius_over_radius(
            std::declval<std::array<double, 3>>(), double()));

    using gradient_return_type_double =
        decltype(std::declval<ConformingType>().gradient(
            std::declval<std::array<double, 3>>()));
    using gradient_return_type_DataVector =
        decltype(std::declval<ConformingType>().gradient(
            std::declval<std::array<DataVector, 3>>()));

    using map_over_radius_return_type_double =
        decltype(std::declval<ConformingType>().map_over_radius(
            std::declval<std::array<double, 3>>()));
    using map_over_radius_return_type_DataVector =
        decltype(std::declval<ConformingType>().map_over_radius(
            std::declval<std::array<DataVector, 3>>()));

    static_assert(std::is_same_v<call_return_type_double, double>,
                  "The 'call' function must return a 'double' when called with "
                  "an array of doubles.");
    static_assert(std::is_same_v<call_return_type_DataVector, DataVector>,
                  "The 'call' function must return a 'DataVector' when called "
                  "with an array of DataVectors.");

    static_assert(
        std::is_same_v<original_radius_over_radius_return_type,
                       std::optional<double>>,
        "The 'original_radius_over_radius' function must return a 'double'.");

    static_assert(
        std::is_same_v<gradient_return_type_double, std::array<double, 3>>,
        "The 'gradient' function must return a 'std::array<double, 3>>' when "
        "called with an array of doubles.");
    static_assert(
        std::is_same_v<gradient_return_type_DataVector,
                       std::array<DataVector, 3>>,
        "The 'gradient' function must return a 'std::array<DataVector, 3>>' "
        "when called with an array of DataVectors.");

    static_assert(std::is_same_v<map_over_radius_return_type_double, double>,
                  "The 'map_over_radius' function must return a 'double' when "
                  "called with an array of doubles.");
    static_assert(
        std::is_same_v<map_over_radius_return_type_DataVector, DataVector>,
        "The 'map_over_radius' function must return a 'DataVector' when called "
        "with an array of DataVectors.");

    static_assert(
        std::is_same_v<bool, decltype(std::declval<ConformingType>() ==
                                      std::declval<ConformingType>())>);
  };
};

}  // namespace domain::CoordinateMaps::protocols
