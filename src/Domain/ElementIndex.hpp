// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ElementIndex.

#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iosfwd>

#include "Utilities/ConstantExpressions.hpp"

/// \cond
template <size_t>
struct ElementId;
/// \endcond
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
}  // namespace ElementIndex_detail

class SegmentIndex {
 public:
  SegmentIndex() noexcept = default;
  SegmentIndex(size_t block_id, const SegmentId& segment_id) noexcept;
  size_t block_id() const noexcept { return block_id_; }
  size_t index() const noexcept { return index_; }
  size_t refinement_level() const noexcept { return refinement_level_; }

 private:
  unsigned block_id_ : ElementIndex_detail::block_id_bits;
  unsigned refinement_level_ : ElementIndex_detail::refinement_bits;
  unsigned index_ : ElementIndex_detail::max_refinement_level;
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& s, const SegmentIndex& index) noexcept;

/// \ingroup ParallelGroup
/// A class for indexing a Charm array by Element.
template <size_t VolumeDim>
class ElementIndex {
 public:
  ElementIndex() = default;
  // clang-tidy: mark explicit: we want to allow conversion
  ElementIndex(const ElementId<VolumeDim>& id) noexcept;  // NOLINT
  size_t block_id() const noexcept { return segments_[0].block_id(); }
  const std::array<SegmentIndex, VolumeDim>& segments() const noexcept {
    return segments_;
  }

 private:
  std::array<SegmentIndex, VolumeDim> segments_;
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

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& s,
                         const ElementIndex<VolumeDim>& index) noexcept;
