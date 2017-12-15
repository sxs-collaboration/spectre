// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Identity.

#pragma once

#include <array>
#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"

namespace CoordinateMaps {

/// \ingroup CoordinateMapsGroup
/// Identity  map from \f$\xi \rightarrow x\f$.
template <size_t Dim>
class Identity {
 public:
  static constexpr size_t dim = Dim;

  Identity() = default;
  ~Identity() = default;
  Identity(const Identity&) = default;
  Identity(Identity&&) noexcept = default;  // NOLINT
  Identity& operator=(const Identity&) = default;
  Identity& operator=(Identity&&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim> operator()(
      const std::array<T, Dim>& source_coords) const;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim> inverse(
      const std::array<T, Dim>& target_coords) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, Dim>& source_coords) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, Dim>& source_coords) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) {}  // NOLINT
};

template <size_t Dim>
inline constexpr bool operator==(
    const CoordinateMaps::Identity<Dim>& /*lhs*/,
    const CoordinateMaps::Identity<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
inline constexpr bool operator!=(
    const CoordinateMaps::Identity<Dim>& lhs,
    const CoordinateMaps::Identity<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace CoordinateMaps
