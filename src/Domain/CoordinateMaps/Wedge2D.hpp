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

/// \ingroup CoordinateMaps
/// Two dimensional map from the unit square to a wedge, which forms one
/// quadrant of an annulus. The wedge can be oriented in the +/- x or y axis,
/// with the inner and outer radii specifiable.
class Wedge2D {
 public:
  static constexpr size_t dim = 2;

  explicit Wedge2D(double inner_radius, double outer_radius,
                   Direction<2> positioning_of_wedge);

  Wedge2D() = default;
  ~Wedge2D() = default;
  Wedge2D(Wedge2D&&) = default;
  Wedge2D& operator=(Wedge2D&&) = default;
  Wedge2D(const Wedge2D&) = default;
  Wedge2D& operator=(const Wedge2D&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2> operator()(
      const std::array<T, 2>& x) const noexcept;

  // Currently unimplemented, hence the noreturn attribute
  template <typename T>
  [[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 2>
  inverse(const std::array<T, 2>& x) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 2>& xi) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 2>& xi) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT
 private:
  friend bool operator==(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;

  double inner_radius_{};
  double outer_radius_{};
  Direction<2> positioning_of_wedge_{};
};
bool operator!=(const Wedge2D& lhs, const Wedge2D& rhs) noexcept;
}  // namespace CoordinateMaps
