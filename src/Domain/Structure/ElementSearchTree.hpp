// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <cstddef>
#include <map>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"

namespace boost::geometry::traits {
// Make Tensor compatible with Boost.Geometry
// This is needed to search for a block-logical coordinate given as a Tensor
// in the `ElementSearchTree`.
template <typename DataType, size_t Dim, typename Frame>
struct tag<tnsr::I<DataType, Dim, Frame>> {
  using type = boost::geometry::point_tag;
};
template <typename DataType, size_t Dim, typename Frame>
struct coordinate_type<tnsr::I<DataType, Dim, Frame>> {
  using type = DataType;
};
template <typename DataType, size_t Dim, typename Frame>
struct coordinate_system<tnsr::I<DataType, Dim, Frame>> {
  using type = boost::geometry::cs::cartesian;
};
template <typename DataType, size_t Dim, typename Frame>
struct dimension<tnsr::I<DataType, Dim, Frame>>
    : std::integral_constant<size_t, Dim> {};
template <typename DataType, size_t Dim, typename Frame, size_t GetDimension>
struct access<tnsr::I<DataType, Dim, Frame>, GetDimension> {
  static constexpr DataType get(const tnsr::I<DataType, Dim, Frame>& point) {
    return ::get<GetDimension>(point);
  }
  static void set(tnsr::I<DataType, Dim, Frame>& point, const DataType& value) {
    ::get<GetDimension>(point) = value;
  }
};
// Make ElementId compatible with Boost.Geometry
// Each ElementId defines a bounding box in block-logical coordinates, which is
// used to search for elements in the `ElementSearchTree`.
template <size_t Dim>
struct tag<ElementId<Dim>> {
  using type = boost::geometry::box_tag;
};
template <size_t Dim>
struct coordinate_type<ElementId<Dim>> {
  using type = double;
};
template <size_t Dim>
struct coordinate_system<ElementId<Dim>> {
  using type = boost::geometry::cs::cartesian;
};
template <size_t Dim>
struct dimension<ElementId<Dim>> : std::integral_constant<size_t, Dim> {};
template <size_t Dim>
struct point_type<ElementId<Dim>> {
  using type = tnsr::I<double, Dim, ::Frame::BlockLogical>;
};
template <size_t Dim, size_t Index, size_t GetDimension>
struct indexed_access<ElementId<Dim>, Index, GetDimension> {
  static constexpr double get(const ElementId<Dim>& element_id) {
    if constexpr (Index == 0) {
      return element_id.segment_id(GetDimension).endpoint(Side::Lower);
    } else {
      return element_id.segment_id(GetDimension).endpoint(Side::Upper);
    }
  }
};
}  // namespace boost::geometry::traits

namespace domain {

/*!
 * \brief Search tree for efficiently looking up elements by their bounding
 * boxes in block-logical coordinates.
 *
 * The search tree is constructed from a list of `ElementId`s, which define
 * bounding boxes in block-logical coordinates. All `ElementId`s must be in the
 * same block. Then, the search tree can be used to efficiently search for the
 * element that contains a given block-logical coordinate (see e.g.
 * `element_logical_coordinates`).
 *
 * Example usage:
 * \snippet Test_ElementSearchTree.cpp element_search_tree_example
 *
 * Use `domain::index_element_ids()` to create one search tree for each block in
 * the domain given the full list of `ElementId`s.
 *
 * \details The search tree is a `boost::geometry::rtree` with a choice of
 * quadratic splitting algorithm and a maximum of 16 elements per node. These
 * choices work well, but haven't been extensively tuned.
 */
template <size_t Dim>
using ElementSearchTree =
    boost::geometry::index::rtree<ElementId<Dim>,
                                  boost::geometry::index::quadratic<16>>;

/*!
 * \brief Sorts element IDs into one `ElementSearchTree` per block for efficient
 * searching.
 *
 * Returns a map of search trees indexed by block ID. Each search tree contains
 * all the `ElementId`s for that block.
 *
 * \see ElementSearchTree
 */
template <size_t Dim>
std::map<size_t, ElementSearchTree<Dim>> index_element_ids(
    const std::vector<ElementId<Dim>>& element_ids) {
  std::map<size_t, ElementSearchTree<Dim>> trees{};
  for (const auto& element_id : element_ids) {
    trees[element_id.block_id()].insert(element_id);
  }
  return trees;
}

}  // namespace domain
