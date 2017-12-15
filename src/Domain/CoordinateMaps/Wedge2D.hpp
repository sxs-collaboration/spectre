// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Wedge2D.

#pragma once

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"

namespace CoordinateMaps {

/// \ingroup CoordinateMapsGroup
/// Two dimensional map from the unit square to a wedge, which is constructed
/// by interpolating between a circular arc of radius `radius_of_circle` and a
/// flat face which is circumscribed by a circular arc of radius
/// `radius_of_other`. These arcs extend \f$\pi/2\f$ in angle, and can be
/// oriented along the +/- x or y axis. The choice of using either equiangular
/// or equidistant coordinates along the arcs is specifiable with
/// `with_equiangular_map`. For a more detailed discussion, see the
/// documentation for Wedge3D.
class Wedge2D {
 public:
  static constexpr size_t dim = 2;

  Wedge2D(double radius_of_other, double radius_of_circle,
          Direction<2> direction_of_wedge, bool with_equiangular_map) noexcept;

  Wedge2D() = default;
  ~Wedge2D() = default;
  Wedge2D(Wedge2D&&) = default;
  Wedge2D& operator=(Wedge2D&&) = default;
  Wedge2D(const Wedge2D&) = default;
  Wedge2D& operator=(const Wedge2D&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> operator()(
      const std::array<T, 2>& source_coords) const noexcept;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> inverse(
      const std::array<T, 2>& target_coords) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 2>& source_coords) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 2>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT
 private:
  friend bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;

  double radius_of_other_{};
  double radius_of_circle_{};
  Direction<2> direction_of_wedge_{};
  bool with_equiangular_map_ = false;
};
bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;
}  // namespace CoordinateMaps
