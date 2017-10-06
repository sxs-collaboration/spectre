// Distributed under the MIT License.
// See LICENSE.txt for details.

/// Defines the class Trilinear.

#pragma once

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"

namespace CoordinateMaps {

/// \ingroup CoordinateMaps
///
/// Trilinear map from the cube to a hexahedron, defined by eight
/// vertices.
class Trilinear {
 public:
  static constexpr size_t dim = 3;
  explicit Trilinear(std::array<std::array<double, 3>, 8> vertices) noexcept;
  Trilinear() = default;
  ~Trilinear() = default;
  Trilinear(Trilinear&&) = default;
  Trilinear(const Trilinear&) = default;
  Trilinear& operator=(const Trilinear&) = default;
  Trilinear& operator=(Trilinear&&) = default;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> operator()(
      const std::array<T, 3>& x) const noexcept;

  // Currently unimplemented, hence the noreturn attribute
  template <typename T>
  [[noreturn]] std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
  inverse(const std::array<T, 3>& x) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, 3>& xi) const noexcept;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, 3>& xi) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

 private:
  friend bool operator==(const Trilinear& lhs, const Trilinear& rhs) noexcept;
  std::array<std::array<double, 3>, 8> vertices_;
};

bool operator!=(const Trilinear& lhs, const Trilinear& rhs) noexcept;
}  // namespace CoordinateMaps
