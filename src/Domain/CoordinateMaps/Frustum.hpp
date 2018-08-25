// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace CoordinateMaps {

/// \ingroup CoordinateMapsGroup
///
/// Frustum map from the cube to a rectangular frustum where the bases of the
/// frustum are perpendicular to the z-axis. The `lower_bound` and
/// `upper_bound` values correspond to the locations of the bases along this
/// axis. The `face_vertices` values determine the size of the two rectangular
/// bases along the other two axes. For example, for a frustum along the z-axis,
/// with the lower base starting at (x,y) = (-2.0,3.0) and extending to
/// (2.0,5.0), and with the upper base extending from (0.0,1.0) to (1.0,3.0),
/// the corresponding value for `face_vertices` is `{{{{-2.0,3.0}}, {{2.0,5.0}},
/// {{0.0,1.0}}, {{1.0,3.0}}}}`. The user may reorient the frustum by passing
/// an `OrientationMap` to the constructor. If `with_equiangular_map` is true,
/// then this coordinate map applies a tangent function mapping to the logical
/// xi and eta coordinates.
class Frustum {
 public:
  static constexpr size_t dim = 3;
  Frustum(const std::array<std::array<double, 2>, 4>& face_vertices,
          double lower_bound, double upper_bound,
          OrientationMap<3> orientation_of_frustum,
          bool with_equiangular_map = false) noexcept;
  Frustum() = default;
  ~Frustum() = default;
  Frustum(Frustum&&) = default;
  Frustum(const Frustum&) = default;
  Frustum& operator=(const Frustum&) = default;
  Frustum& operator=(Frustum&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

  /// Returns boost::none if \f$z\f$ is at or beyond the \f$z\f$-coordinate of
  /// the apex of the pyramid, tetrahedron, or triangular prism that is
  /// formed by extending the `Frustum` (for a
  /// \f$z\f$-oriented `Frustum`).
  boost::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const noexcept;

  // clang-tidy: google runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  friend bool operator==(const Frustum& lhs, const Frustum& rhs) noexcept;

  OrientationMap<3> orientation_of_frustum_{};
  double sum_midpoint_x_{std::numeric_limits<double>::signaling_NaN()};
  double dif_midpoint_x_{std::numeric_limits<double>::signaling_NaN()};
  double sum_half_length_x_{std::numeric_limits<double>::signaling_NaN()};
  double dif_half_length_x_{std::numeric_limits<double>::signaling_NaN()};
  double sum_midpoint_y_{std::numeric_limits<double>::signaling_NaN()};
  double dif_midpoint_y_{std::numeric_limits<double>::signaling_NaN()};
  double sum_half_length_y_{std::numeric_limits<double>::signaling_NaN()};
  double dif_half_length_y_{std::numeric_limits<double>::signaling_NaN()};
  double midpoint_z_{std::numeric_limits<double>::signaling_NaN()};
  double half_length_z_{std::numeric_limits<double>::signaling_NaN()};
  bool with_equiangular_map_{false};
};

bool operator!=(const Frustum& lhs, const Frustum& rhs) noexcept;
}  // namespace CoordinateMaps
