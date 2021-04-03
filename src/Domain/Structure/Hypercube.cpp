// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/Hypercube.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <ostream>

#include "Domain/Structure/Side.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElement<ElementDim, HypercubeDim>::HypercubeElement(
    std::array<size_t, ElementDim> dimensions_in_parent,
    std::array<Side, HypercubeDim - ElementDim> index) noexcept
    : dimensions_in_parent_{std::move(dimensions_in_parent)},
      index_{std::move(index)} {
  ASSERT(
      not std::any_of(
          dimensions_in_parent_.begin(), dimensions_in_parent_.end(),
          [](const size_t d) noexcept { return d >= HypercubeDim; }),
      "Found dimension that exceeds the hypercube dimension in construction of "
          << HypercubeDim << "D element: " << dimensions_in_parent_);
  if constexpr (ElementDim > 1) {
    std::sort(dimensions_in_parent_.begin(), dimensions_in_parent_.end());
    ASSERT(
        [this]() noexcept {
          for (size_t d = 1; d < ElementDim; ++d) {
            if (gsl::at(dimensions_in_parent_, d) ==
                gsl::at(dimensions_in_parent_, d - 1)) {
              return false;
            }
          }
          return true;
        }(),
        "Found repeated dimension in construction of hypercube element: "
            << dimensions_in_parent_ << " (sorted)");
  }
}

template <size_t ElementDim, size_t HypercubeDim>
template <size_t LocalElementDim, Requires<LocalElementDim == 0>>
HypercubeElement<ElementDim, HypercubeDim>::HypercubeElement(
    std::array<Side, HypercubeDim> index) noexcept
    : index_{std::move(index)} {}

template <size_t ElementDim, size_t HypercubeDim>
template <size_t LocalElementDim, Requires<LocalElementDim == 1>>
HypercubeElement<ElementDim, HypercubeDim>::HypercubeElement(
    size_t dim_in_parent, std::array<Side, HypercubeDim - 1> index) noexcept
    : dimensions_in_parent_{dim_in_parent}, index_{std::move(index)} {
  ASSERT(dim_in_parent < HypercubeDim,
         "Dimension " << dim_in_parent << " exceeds hypercube dimension in "
                      << HypercubeDim << "D");
}

template <size_t ElementDim, size_t HypercubeDim>
const std::array<size_t, ElementDim>&
HypercubeElement<ElementDim, HypercubeDim>::dimensions_in_parent() const
    noexcept {
  return dimensions_in_parent_;
}

template <size_t ElementDim, size_t HypercubeDim>
template <size_t LocalElementDim, Requires<LocalElementDim == 1>>
size_t HypercubeElement<ElementDim, HypercubeDim>::dimension_in_parent() const
    noexcept {
  return dimensions_in_parent_[0];
}

template <size_t ElementDim, size_t HypercubeDim>
const std::array<Side, HypercubeDim - ElementDim>&
HypercubeElement<ElementDim, HypercubeDim>::index() const noexcept {
  return index_;
}

template <size_t ElementDim, size_t HypercubeDim>
const Side&
HypercubeElement<ElementDim, HypercubeDim>::side_in_parent_dimension(
    size_t d) const noexcept {
  ASSERT(not std::any_of(dimensions_in_parent_.begin(),
                         dimensions_in_parent_.end(),
                         [&d](const size_t dim_in_parent) noexcept {
                           return dim_in_parent == d;
                         }),
         "The parent dimension "
             << d << " is aligned with the hypercube element '" << *this
             << "', so the element is not located at a particular side in this "
                "dimension.");
  for (size_t dim_in_element = ElementDim; dim_in_element > 0;
       --dim_in_element) {
    if (gsl::at(dimensions_in_parent_, dim_in_element - 1) <= d) {
      --d;
    }
  }
  return gsl::at(index_, d);
}

template <size_t ElementDim, size_t HypercubeDim>
std::ostream& operator<<(
    std::ostream& os,
    const HypercubeElement<ElementDim, HypercubeDim>& element) noexcept {
  if constexpr (ElementDim == 0) {
    os << "Vertex";
  } else if constexpr (ElementDim == 1) {
    os << "Edge";
  } else if constexpr (ElementDim == 2) {
    os << "Face";
  } else if constexpr (ElementDim == 3) {
    os << "Cell";
  } else {
    os << ElementDim << "-face";
  }
  return os << HypercubeDim << "D[" << element.dimensions_in_parent() << ","
            << element.index() << "]";
}

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElementsIterator<ElementDim,
                          HypercubeDim>::HypercubeElementsIterator() noexcept
    : index_{0} {
  if constexpr (ElementDim > 0) {
    for (size_t d = 0; d < ElementDim; ++d) {
      gsl::at(dimensions_in_parent_, d) = d;
    }
  }
}

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>::begin() noexcept {
  return {};
}

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>::end() noexcept {
  HypercubeElementsIterator end_iterator{};
  if constexpr (ElementDim > 0) {
    for (size_t d = 0; d < ElementDim; ++d) {
      gsl::at(end_iterator.dimensions_in_parent_, d) = num_indices + d + 1;
    }
  } else {
    end_iterator.index_ = two_to_the(num_indices);
  }
  return end_iterator;
}

template <size_t ElementDim, size_t HypercubeDim>
template <size_t LocalElementDim, Requires<(LocalElementDim > 0)>>
void HypercubeElementsIterator<ElementDim, HypercubeDim>::
    increment_dimension_in_parent(const size_t d) noexcept {
  ++gsl::at(dimensions_in_parent_, d);
  if (gsl::at(dimensions_in_parent_, d) == num_indices + d + 1 and d > 0) {
    increment_dimension_in_parent(d - 1);
    gsl::at(dimensions_in_parent_, d) =
        gsl::at(dimensions_in_parent_, d - 1) + 1;
  }
}

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>&
HypercubeElementsIterator<ElementDim, HypercubeDim>::operator++() noexcept {
  ++index_;
  if constexpr (ElementDim > 0) {
    if (index_ == two_to_the(num_indices)) {
      index_ = 0;
      increment_dimension_in_parent(ElementDim - 1);
    }
  }
  return *this;
}

template <size_t ElementDim, size_t HypercubeDim>
// NOLINTNEXTLINE(cert-dcl21-cpp) see declaration
HypercubeElementsIterator<ElementDim, HypercubeDim>
HypercubeElementsIterator<ElementDim, HypercubeDim>::operator++(int) noexcept {
  const auto ret = *this;
  operator++();
  return ret;
}

template <size_t ElementDim, size_t HypercubeDim>
HypercubeElement<ElementDim, HypercubeDim>
    HypercubeElementsIterator<ElementDim, HypercubeDim>::operator*() const
    noexcept {
  const std::bitset<num_indices> index_bits{index_};
  std::array<Side, num_indices> sides{};
  for (size_t d = 0; d < num_indices; ++d) {
    gsl::at(sides, d) = index_bits[d] ? Side::Upper : Side::Lower;
  }
  return HypercubeElement<ElementDim, HypercubeDim>{dimensions_in_parent_,
                                                    std::move(sides)};
}

template <size_t ElementDim, size_t HypercubeDim>
bool operator==(
    const HypercubeElementsIterator<ElementDim, HypercubeDim>& lhs,
    const HypercubeElementsIterator<ElementDim, HypercubeDim>& rhs) noexcept {
  return lhs.dimensions_in_parent_ == rhs.dimensions_in_parent_ and
         lhs.index_ == rhs.index_;
}

template <size_t ElementDim, size_t HypercubeDim>
bool operator!=(
    const HypercubeElementsIterator<ElementDim, HypercubeDim>& lhs,
    const HypercubeElementsIterator<ElementDim, HypercubeDim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(ELEMENT_DIM, HYPERCUBE_DIM)                              \
  template struct HypercubeElement<ELEMENT_DIM, HYPERCUBE_DIM>;              \
  template std::ostream& operator<<(                                         \
      std::ostream& os,                                                      \
      const HypercubeElement<ELEMENT_DIM, HYPERCUBE_DIM>& element) noexcept; \
  template struct HypercubeElementsIterator<ELEMENT_DIM, HYPERCUBE_DIM>;     \
  template bool operator==(                                                  \
      const HypercubeElementsIterator<ELEMENT_DIM, HYPERCUBE_DIM>& lhs,      \
      const HypercubeElementsIterator<ELEMENT_DIM, HYPERCUBE_DIM>&           \
          rhs) noexcept;                                                     \
  template bool operator!=(                                                  \
      const HypercubeElementsIterator<ELEMENT_DIM, HYPERCUBE_DIM>& lhs,      \
      const HypercubeElementsIterator<ELEMENT_DIM, HYPERCUBE_DIM>&           \
          rhs) noexcept;
#define INSTANTIATE_VERTEX(r, data)                          \
  INSTANTIATE(0, DIM(data))                                  \
  template HypercubeElement<0, DIM(data)>::HypercubeElement( \
      std::array<Side, DIM(data)>);
#define INSTANTIATE_EDGE(r, data)                                       \
  INSTANTIATE(1, DIM(data))                                             \
  template HypercubeElement<1, DIM(data)>::HypercubeElement(            \
      size_t, std::array<Side, DIM(data) - 1>);                         \
  template size_t HypercubeElement<1, DIM(data)>::dimension_in_parent() \
      const noexcept;
#define INSTANTIATE_FACE(r, data) INSTANTIATE(2, DIM(data))
#define INSTANTIATE_CELL(r, data) INSTANTIATE(3, DIM(data))

GENERATE_INSTANTIATIONS(INSTANTIATE_VERTEX, (0, 1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_EDGE, (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_FACE, (2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATE_CELL, (3))

#undef DIM
#undef INSTANTIATE
#undef INSTANTIATE_VERTEX
#undef INSTANTIATE_EDGE
#undef INSTANTIATE_FACE
#undef INSTANTIATE_CELL
