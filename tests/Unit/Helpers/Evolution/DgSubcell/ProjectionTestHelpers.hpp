// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Index;
template <size_t Dim>
class Mesh;
/// \endcond

namespace TestHelpers::evolution::dg::subcell {
// computes a simple polynomial over the grid that we then project and
// reconstruct in the tests.
template <size_t Dim, typename Fr>
DataVector cell_values(size_t max_polynomial_degree_plus_one,
                       const tnsr::I<DataVector, Dim, Fr>& coords);

// Computes the average in each finite volume cell multiplied by the cell's
// volume.
template <size_t Dim>
DataVector cell_averages_times_volume(size_t max_polynomial_degree_plus_one,
                                      const Index<Dim>& subcell_extents);
}  // namespace TestHelpers::evolution::dg::subcell
