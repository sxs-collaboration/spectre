// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Redistributes gridpoints on the sphere.
 * \image html EquatorialCompression.png "A sphere with an `aspect_ratio` of 3."
 *
 * \details A mapping from the sphere to itself which redistributes points
 * towards (or away from) a user-specifed axis, indicated by `index_pole_axis_`.
 * Once the axis is selected, the map is determined by a single parameter,
 * the `aspect_ratio` \f$\alpha\f$, which is the ratio of the distance
 * perpendicular to the polar axis to the distance along the polar axis for
 * a given point. This parameter name was chosen because points with
 * \f$\tan \theta = 1\f$ get mapped to points with \f$\tan \theta' = \alpha\f$.
 * In general, gridpoints located at an angle \f$\theta\f$ from the pole are
 * mapped to a new angle
 * \f$\theta'\f$ satisfying \f$\tan \theta' = \alpha \tan \theta\f$.
 *
 * For an `aspect_ratio` greater than one, the gridpoints are mapped towards
 * the equator, leading to an equatorially compressed grid. For an
 * `aspect_ratio` less than one, the gridpoints are mapped towards the poles.
 * Note that the aspect ratio must be positive.
 *
 * Suppose the polar axis were the z-axis, given by `index_pole_axis_ == 2`.
 * We can then define the auxiliary variables \f$ r := \sqrt{x^2 + y^2 +z^2}\f$
 * and \f$ \rho := \sqrt{x^2 + y^2 + \alpha^{-2} z^2}\f$.
 *
 * The map corresponding to this transformation in cartesian coordinates
 * is then given by:
 *
 * \f[\vec{x}'(x,y,z) =
 * \frac{r}{\rho}\begin{bmatrix}
 * x\\
 * y\\
 * \alpha^{-1} z\\
 * \end{bmatrix}.\f]
 *
 * The mappings for polar axes along the x and y axes are similarly obtained.
 */
class EquatorialCompression {
 public:
  static constexpr size_t dim = 3;
  explicit EquatorialCompression(double aspect_ratio,
                                 size_t index_pole_axis = 2);
  EquatorialCompression() = default;
  ~EquatorialCompression() = default;
  EquatorialCompression(EquatorialCompression&&) = default;
  EquatorialCompression(const EquatorialCompression&) = default;
  EquatorialCompression& operator=(const EquatorialCompression&) = default;
  EquatorialCompression& operator=(EquatorialCompression&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // clang-tidy: google runtime references
  void pup(PUP::er& p);  // NOLINT

  bool is_identity() const { return is_identity_; }

 private:
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> angular_distortion(
      const std::array<T, 3>& coords, double inverse_alpha) const;
  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
  angular_distortion_jacobian(const std::array<T, 3>& coords,
                              double inverse_alpha) const;
  friend bool operator==(const EquatorialCompression& lhs,
                         const EquatorialCompression& rhs);

  double aspect_ratio_{std::numeric_limits<double>::signaling_NaN()};
  double inverse_aspect_ratio_{std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
  size_t index_pole_axis_{};
};
bool operator!=(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs);
}  // namespace CoordinateMaps
}  // namespace domain
