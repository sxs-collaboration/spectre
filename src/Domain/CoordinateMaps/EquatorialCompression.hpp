// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace domain {
namespace CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Redistributes gridpoints on the sphere.
 * \image html EquatorialCompression.png "A sphere with an `aspect_ratio` of 3."
 *
 * \details A mapping from the sphere to itself which depends on a single
 * parameter, the `aspect_ratio` \f$\alpha\f$, which is the ratio of the
 * horizontal length to the vertical height for a given point. This parameter
 * name was chosen because points with \f$\tan \theta = 1\f$ get mapped to
 * points with \f$\tan \theta' = \alpha\f$. In general, gridpoints located
 * at an angle \f$\theta\f$ from the pole are mapped to a new angle
 * \f$\theta'\f$ satisfying \f$\tan \theta' = \alpha \tan \theta\f$.
 *
 * For an `aspect_ratio` greater than one, the gridpoints are mapped towards
 * the equator, leading to an equatorially compressed grid. For an
 * `aspect_ratio` less than one, the gridpoints are mapped towards the poles.
 * Note that the aspect ratio must be positive.
 *
 * We define the auxiliary variables \f$ r := \sqrt{x^2 + y^2 +z^2}\f$
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
 * \end{bmatrix}\f]
 *
 */
class EquatorialCompression {
 public:
  static constexpr size_t dim = 3;
  explicit EquatorialCompression(double aspect_ratio) noexcept;
  EquatorialCompression() = default;
  ~EquatorialCompression() = default;
  EquatorialCompression(EquatorialCompression&&) = default;
  EquatorialCompression(const EquatorialCompression&) = default;
  EquatorialCompression& operator=(const EquatorialCompression&) = default;
  EquatorialCompression& operator=(EquatorialCompression&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const noexcept;

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

  bool is_identity() const noexcept { return is_identity_; }

 private:
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> angular_distortion(
      const std::array<T, 3>& coords, double inverse_alpha) const noexcept;
  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame>
  angular_distortion_jacobian(const std::array<T, 3>& coords,
                              double inverse_alpha) const noexcept;
  friend bool operator==(const EquatorialCompression& lhs,
                         const EquatorialCompression& rhs) noexcept;

  double aspect_ratio_{std::numeric_limits<double>::signaling_NaN()};
  double inverse_aspect_ratio_{std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
};
bool operator!=(const EquatorialCompression& lhs,
                const EquatorialCompression& rhs) noexcept;
}  // namespace CoordinateMaps
}  // namespace domain
