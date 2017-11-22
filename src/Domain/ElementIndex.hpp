// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ElementIndex.

#pragma once

#include <array>
#include <iosfwd>

#include "Utilities/ConstantExpressions.hpp"

template <size_t>
struct ElementId;
struct SegmentId;

namespace ElementIndex_detail {
constexpr size_t block_id_bits = 7;
constexpr size_t refinement_bits = 5;
constexpr size_t max_refinement_level = 20;
static_assert(block_id_bits + refinement_bits + max_refinement_level ==
              8 * sizeof(int),
              "Bit representation requires padding or is too large");
static_assert(two_to_the(refinement_bits) >= max_refinement_level,
              "Not enough bits to represent all refinement levels");

class SegmentIndex {
 public:
  SegmentIndex() = default;
  SegmentIndex(size_t block_id, const SegmentId& segment_id) noexcept;
 private:
  unsigned block_id_ : block_id_bits;
  unsigned refinement_level_ : refinement_bits;
  unsigned index_ : max_refinement_level;

  friend std::ostream& operator<<(std::ostream& s,
                                  const SegmentIndex& index) noexcept;
};
}  // namespace ElementIndex_detail

template <size_t>
class ElementIndex;

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& s,
                         const ElementIndex<VolumeDim>& index) noexcept;

/// \ingroup ParallelGroup
/// A class for indexing a Charm array by Element.
template <size_t VolumeDim>
class ElementIndex {
 public:
  ElementIndex() = default;
  explicit ElementIndex(const ElementId<VolumeDim>& id) noexcept;
 private:
  std::array<ElementIndex_detail::SegmentIndex, VolumeDim> segments_;

  friend std::ostream& operator<<<VolumeDim>(
      std::ostream& s, const ElementIndex<VolumeDim>& index) noexcept;
};

template <size_t VolumeDim>
bool operator==(const ElementIndex<VolumeDim>& a,
                const ElementIndex<VolumeDim>& b) noexcept;
template <size_t VolumeDim>
bool operator!=(const ElementIndex<VolumeDim>& a,
                const ElementIndex<VolumeDim>& b) noexcept;

template <size_t VolumeDim>
size_t hash_value(const ElementIndex<VolumeDim>& index) noexcept;

namespace std {
template <size_t VolumeDim>
struct hash<ElementIndex<VolumeDim>> {
  size_t operator()(const ElementIndex<VolumeDim>& x) const noexcept;
};
}  // namespace std
