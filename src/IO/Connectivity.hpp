// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for computing the connectivity of an element

#pragma once

#include <cstddef>
#include <ostream>
#include <utility>
#include <vector>

template <size_t Dim>
class Index;

/// Holds functions needed for visualizing data
namespace vis {
namespace detail {
/*!
 * \brief A list of all topologies for which we can compute the number of cells
 */
enum class Topology { Line, Quad, Hexahedron };

std::ostream& operator<<(std::ostream& os, const Topology& topology);

/*!
 * \brief Represents the number of cells in a particular topology
 *
 * Each `CellInTopology` holds an enum of type `Topology` whose
 * value denotes the type of the topology, e.g. line, quad or hexahedron, and a
 * vector of bounding indices which are the indices of the grid coordinates in
 * the contiguous arrays of x, y, and z coordinates that bound the cell.
 */
struct CellInTopology {
  // cppcheck-suppress passedByValue
  CellInTopology(const Topology& top, std::vector<size_t> bounding_ind)
      : topology(top), bounding_indices(std::move(bounding_ind)) {}
  CellInTopology() = default;
  CellInTopology(const CellInTopology& /*rhs*/) = default;
  CellInTopology(CellInTopology&& /*rhs*/) = default;
  CellInTopology& operator=(const CellInTopology& /*rhs*/) = default;
  CellInTopology& operator=(CellInTopology&& /*rhs*/) = default;
  ~CellInTopology() = default;
  Topology topology{Topology::Line};
  std::vector<size_t> bounding_indices{};
};

// @{
/*!
 * \brief Compute the cells in the element.
 *
 * Returns a vector of the cells in the topology I1^Dim, i.e. a line if Dim ==
 * 1, or a hexahedron if Dim == 3. The cells are bounded by lines connecting
 * grid points along the axes of the element, so if you have (n_x by n_y by n_z)
 * grid points, you have ((n_x-1) by (n_y-1) by (n_z-1)) cells.
 *
 * \note As more topologies are added, e.g. S2, the interface will need slight
 * modification, however the return type is likely to be able to remain the
 * same.
 */
template <size_t Dim>
std::vector<CellInTopology> compute_cells(const Index<Dim>& extents) noexcept;

std::vector<CellInTopology> compute_cells(
    const std::vector<size_t>& extents) noexcept;
// @}
}  // namespace detail
}  // namespace vis
