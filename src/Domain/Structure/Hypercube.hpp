// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <ostream>

#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"

/*!
 * \brief An element of dimension `ElementDim` on the boundary of a hypercube of
 * dimension `HypercubeDim`
 *
 * A hypercube of dimension \f$n\f$ (`HypercubeDim`) is composed of
 * \f$2^{n-k}\binom{n}{k}\f$ elements of dimension \f$k \leq n\f$
 * (`ElementDim`). For example, a 3D cube has 8 vertices (\f$k=0\f$), 12 edges
 * (\f$k=1\f$), 6 faces (\f$k=2\f$) and 1 cell (\f$k=3\f$). Each element is
 * identified by the \f$k\f$ dimensions it shares with the parent hypercube
 * and \f$n - k\f$ indices that specify whether it is located on the lower or
 * upper side of the parent hypercube's remaining dimensions.
 */
template <size_t ElementDim, size_t HypercubeDim>
struct HypercubeElement {
  static_assert(ElementDim <= HypercubeDim);

  HypercubeElement(std::array<size_t, ElementDim> dimensions_in_parent,
                   std::array<Side, HypercubeDim - ElementDim> index);

  template <size_t NumIndices = HypercubeDim - ElementDim,
            Requires<NumIndices == 0> = nullptr>
  HypercubeElement() {
    for (size_t d = 0; d < ElementDim; ++d) {
      gsl::at(dimensions_in_parent_, d) = d;
    }
  }

  template <size_t LocalElementDim = ElementDim,
            Requires<LocalElementDim == 0> = nullptr>
  explicit HypercubeElement(std::array<Side, HypercubeDim> index);

  template <typename... Indices, size_t LocalElementDim = ElementDim,
            size_t LocalHypercubeDim = HypercubeDim,
            Requires<(LocalElementDim == 0 and LocalHypercubeDim > 0 and
                      sizeof...(Indices) == LocalHypercubeDim)> = nullptr>
  explicit HypercubeElement(Indices... indices)
      : index_{{static_cast<Side>(indices)...}} {}

  template <size_t LocalElementDim = ElementDim,
            Requires<LocalElementDim == 1> = nullptr>
  HypercubeElement(size_t dim_in_parent,
                   std::array<Side, HypercubeDim - 1> index);

  /// @{
  /// The parent hypercube's dimensions that this element shares
  const std::array<size_t, ElementDim>& dimensions_in_parent() const;

  template <size_t LocalElementDim = ElementDim,
            Requires<LocalElementDim == 1> = nullptr>
  size_t dimension_in_parent() const;
  /// @}

  /// @{
  /// Whether this element is located on the lower or upper side in those
  /// dimensions that it does not share with its parent hypercube
  const std::array<Side, HypercubeDim - ElementDim>& index() const;

  const Side& side_in_parent_dimension(size_t d) const;

  template <size_t NumIndices = HypercubeDim - ElementDim,
            Requires<NumIndices == 1> = nullptr>
  const Side& side() const {
    return index_[0];
  }
  /// @}

  bool operator==(const HypercubeElement& rhs) const {
    return dimensions_in_parent_ == rhs.dimensions_in_parent_ and
           index_ == rhs.index_;
  }

  bool operator!=(const HypercubeElement& rhs) const {
    return not(*this == rhs);
  }

 private:
  std::array<size_t, ElementDim> dimensions_in_parent_{};
  std::array<Side, HypercubeDim - ElementDim> index_{};
};

template <size_t ElementDim, size_t HypercubeDim>
std::ostream& operator<<(
    std::ostream& os,
    const HypercubeElement<ElementDim, HypercubeDim>& element);

/// A vertex in a `Dim`-dimensional hypercube
template <size_t Dim>
using Vertex = HypercubeElement<0, Dim>;

/// An edge in a `Dim`-dimensional hypercube
template <size_t Dim>
using Edge = HypercubeElement<1, Dim>;

/// A face in a `Dim`-dimensional hypercube
template <size_t Dim>
using Face = HypercubeElement<2, Dim>;

/// A cell in a `Dim`-dimensional hypercube
template <size_t Dim>
using Cell = HypercubeElement<3, Dim>;

/*!
 * \brief Iterator over all `ElementDim`-dimensional elements on the boundary of
 * a `HypercubeDim`-dimensional hypercube.
 *
 * \see `HypercubeElement`
 */
template <size_t ElementDim, size_t HypercubeDim>
struct HypercubeElementsIterator {
  static_assert(
      ElementDim <= HypercubeDim,
      "Hypercube element dimension must not exceed hypercube dimension.");

 public:
  static constexpr size_t num_indices = HypercubeDim - ElementDim;

  /// The number of `ElementDim`-dimensional elements on the boundary of a
  /// `HypercubeDim`-dimensional hypercube.
  static constexpr size_t size() {
    return two_to_the(num_indices) * factorial(HypercubeDim) /
           factorial(ElementDim) / factorial(num_indices);
  }

  HypercubeElementsIterator();

  static HypercubeElementsIterator begin();

  static HypercubeElementsIterator end();

  HypercubeElementsIterator& operator++();

  // NOLINTNEXTLINE(cert-dcl21-cpp) returned object doesn't need to be const
  HypercubeElementsIterator operator++(int);

  HypercubeElement<ElementDim, HypercubeDim> operator*() const;

 private:
  template <size_t LocalElementDim = ElementDim,
            Requires<(LocalElementDim > 0)> = nullptr>
  void increment_dimension_in_parent(size_t d);

  template <size_t LocalElementDim, size_t LocalHypercubeDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(
      const HypercubeElementsIterator<LocalElementDim, LocalHypercubeDim>& lhs,
      const HypercubeElementsIterator<LocalElementDim, LocalHypercubeDim>& rhs);

  std::array<size_t, ElementDim> dimensions_in_parent_{};
  size_t index_ = std::numeric_limits<size_t>::max();
};

template <size_t ElementDim, size_t HypercubeDim>
bool operator!=(const HypercubeElementsIterator<ElementDim, HypercubeDim>& lhs,
                const HypercubeElementsIterator<ElementDim, HypercubeDim>& rhs);

/// Iterate over all vertices in a `Dim`-dimensional hypercube
template <size_t Dim>
using VertexIterator = HypercubeElementsIterator<0, Dim>;

/// Iterate over all edges in a `Dim`-dimensional hypercube
template <size_t Dim>
using EdgeIterator = HypercubeElementsIterator<1, Dim>;

/// Iterate over all faces in a `Dim`-dimensional hypercube
template <size_t Dim>
using FaceIterator = HypercubeElementsIterator<2, Dim>;

/// Iterate over all cells in a `Dim`-dimensional hypercube
template <size_t Dim>
using CellIterator = HypercubeElementsIterator<3, Dim>;
