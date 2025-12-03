// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace domain {
/// \cond
template <size_t Dim>
class ElementSearchTree;
/// \endcond

template <size_t Dim>
class ElementSearchTreeIterator {
 public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = ElementId<Dim>;
  using difference_type = std::ptrdiff_t;
  using pointer = const ElementId<Dim>*;
  using reference = const ElementId<Dim>&;

  ElementSearchTreeIterator();
  ElementSearchTreeIterator(ElementSearchTreeIterator&&);
  ElementSearchTreeIterator(const ElementSearchTreeIterator&);
  ElementSearchTreeIterator& operator=(ElementSearchTreeIterator&&);
  ElementSearchTreeIterator& operator=(const ElementSearchTreeIterator&);
  ~ElementSearchTreeIterator();

  reference operator*() const;
  pointer operator->() const;

  ElementSearchTreeIterator& operator++();
  ElementSearchTreeIterator operator++(int);

 private:
  friend class ElementSearchTree<Dim>;

  template <typename T>
  static ElementSearchTreeIterator from_impl(T impl);

  template <size_t Dim2>
  friend bool operator==(const ElementSearchTreeIterator<Dim2>& a,
                         const ElementSearchTreeIterator<Dim2>& b);

  // Use an array and manual construction rather than a simpler pimpl
  // since iterators are constructed a lot.
  std::array<char, 8> data_;
};

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
class ElementSearchTree {
 public:
  ElementSearchTree();
  ElementSearchTree(ElementSearchTree&&);
  ElementSearchTree& operator=(ElementSearchTree&&);
  ElementSearchTree(const ElementSearchTree&) = delete;
  ElementSearchTree& operator=(const ElementSearchTree&) = delete;
  ~ElementSearchTree();

  /// Construct a search tree containing the ids in `[begin, end)`.
  template <typename Iter>
  ElementSearchTree(Iter begin, const Iter end) : ElementSearchTree() {
    insert(begin, end);
  }

  size_t size() const;
  bool empty() const;
  void clear();

  void insert(const ElementId<Dim>& id);

  /// Insert the ids in `[begin, end)`.
  template <typename Iter>
  void insert(Iter begin, const Iter end) {
    while (begin != end) {
      insert(*begin);
      ++begin;
    }
  }

  ElementSearchTreeIterator<Dim> begin_covers(
      const tnsr::I<double, Dim, Frame::BlockLogical>& coords) const;

  ElementSearchTreeIterator<Dim> end_covers() const;

 private:
  // Use a pimpl to keep all the boost::geometry stuff in a cpp file,
  // because the boost headers are quite expensive to include.
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

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
    const std::vector<ElementId<Dim>>& element_ids);
}  // namespace domain
