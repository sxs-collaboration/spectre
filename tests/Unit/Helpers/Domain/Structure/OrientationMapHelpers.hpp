// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <random>
#include <vector>

/// \cond
template <size_t Dim>
class OrientationMap;
namespace gsl {
template <typename T>
class not_null;
}
/// \endcond

namespace TestHelpers::domain {
/// \brief List of all valid OrientationMap%s in a given dimension
///
/// \details In more than one dimension, an OrientationMap is considered to be
/// valid if it has a positive determinant for its discrete_rotation_jacobian.
/// This restriction is relaxed in one dimension so that a non-aligned
/// OrientationMap can be tested in 1D.
template <size_t Dim>
std::vector<OrientationMap<Dim>> valid_orientation_maps();

/// Return a random sample of unique OrientationMap%s
///
/// \note If the requested number_of_samples is larger than the number of
/// possible OrientationMap%s, this will return all possible OrientationMap%s
template <size_t Dim>
std::vector<OrientationMap<Dim>> random_orientation_maps(
    size_t number_of_samples, gsl::not_null<std::mt19937*> generator);
}  // namespace TestHelpers::domain
