// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps {

/*!
 * \ingroup CoordinateMapsGroup
 *
 * \brief Pfaffian transformation from spherical to Cartesian coordinates.
 *
 * \note This map is designed to be used together with Spherepack! Spherepack
 * expects a Pfaffian transformation of the derivatives.
 *
 * \details This is a Pfaffian mapping from \f$(r,\theta,\phi) \rightarrow
 * (x,y,z) \f$.
 *
 * The formula for the mapping is...
 * \f{eqnarray*}
 *     x &=& r \sin\theta \cos\phi \\
 *     y &=& r \sin\theta \sin\phi \\
 *     z &=& r \cos\theta
 * \f}
 *
 * The Pfaffian basis vectors
 * \f$ (e_{\hat r}, e_{\hat \theta}, e_{\hat \phi})\f$
 * are related to the coordinate basis vectors
 * \f$ (e_r, e_{\theta}, e_{\phi})\f$
 * by...
 * \f{eqnarray*}
 *     e_{\hat r}      &=& e_r \\
 *     e_{\hat \theta} &=& e_{\theta} \\
 *     e_{\hat \phi}   &=& \frac{1}{\sin \theta} e_{\phi}
 * \f}
 */
class SphericalToCartesianPfaffian {
 public:
  static constexpr size_t dim = 3;
  SphericalToCartesianPfaffian();
  ~SphericalToCartesianPfaffian() = default;
  SphericalToCartesianPfaffian(SphericalToCartesianPfaffian&&);
  SphericalToCartesianPfaffian(const SphericalToCartesianPfaffian&);
  SphericalToCartesianPfaffian& operator=(const SphericalToCartesianPfaffian&);
  SphericalToCartesianPfaffian& operator=(SphericalToCartesianPfaffian&&);

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static constexpr bool is_identity() { return false; }

  static constexpr bool supports_hessian{false};
};

bool operator==(const SphericalToCartesianPfaffian& lhs,
                const SphericalToCartesianPfaffian& rhs);

bool operator!=(const SphericalToCartesianPfaffian& lhs,
                const SphericalToCartesianPfaffian& rhs);
}  // namespace domain::CoordinateMaps
