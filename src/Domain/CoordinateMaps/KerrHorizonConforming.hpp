// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain::CoordinateMaps {

/*!
 * \brief Distorts cartesian coordinates \f$x^i\f$ such that a coordinate sphere
 * \f$\delta_{ij}x^ix^j=C\f$ is mapped to an ellipsoid of constant
 * Kerr-Schild radius \f$r=C\f$.
 *
 * The Kerr-Schild radius \f$r\f$ is defined as the largest positive
 * root of
 *
 *  \f{equation*}{
 *  r^4 - r^2 (x^2 - a^2) - (\vec{a}\cdot \vec{x})^2 = 0.
 *  \f}
 *
 * In this equation, \f$\vec{x}\f$ are the coordinates of the (distorted)
 * surface, and \f$\vec{a}=\vec{S}/M\f$ is the spin-parameter of the black hole.
 * This is equivalent to the implicit definition of \f$r\f$ in Eq. (47) of
 * \cite Lovelace2008tw. The black hole is assumed to be at the origin.
 *
 * \details Given a spin vector \f$\vec{a}\f$, we define the Kerr-Schild radius:
 *
 * \f{equation}{r(\vec{x}) = \sqrt{\frac{x^2 - a^2 + \sqrt{(x^2 -
 * a^2)^2 + 4 (\vec{x} \cdot \vec{a})^2}}{2}} \f}
 *
 * We also define the auxiliary variable:
 *
 * \f{equation}{s(\vec{x}) = \frac{x^2(x^2 + a^2)}{x^4 + (\vec{x}
 * \cdot \vec{a})^2} \f}
 *
 * The map is then given by:
 *
 * \f{equation}{\vec{x}(\vec{\xi}) = \vec{\xi} \sqrt{s(\vec{\xi})} \f}
 *
 * The inverse map is given by:
 * \f{equation}{\vec{\xi}(\vec{x}) = \vec{x} \frac{r(\vec{x})}{|\vec{x}|}
 * \f}
 *
 * The jacobian is:
 * \f{equation}{ \frac{\partial \xi^i}{\partial x^j} =
 * \delta_{ij} \sqrt{s(\vec{\xi})} +
 * \frac{\xi_j}{2 \sqrt{s(\vec{\xi})}} \frac{4\xi^2(1 -
 * s(\vec{\xi})) \xi_j + 2a^2\xi_i + 2 s(\vec{\xi})
 * (\vec{\xi} \cdot \vec{a}) a_i}{\xi^4 + (\vec{\xi} \cdot
 * \vec{a})^2} \f}
 *
 * The inverse jacobian is:
 * \f{equation}{\frac{\partial x^i}{\partial \xi^j} =
 * \delta_{ij}\frac{r(\vec{x})}{|\vec{x}|}
 * + \frac{r(\vec{x})^2 x_i + (\vec{x} \cdot \vec{a})a_i}{2r(\vec{x})^3
 * - r(\vec{x}) (x^2 - a^2)}\frac{x_j}{|\vec{x}|} - \frac{r(\vec{x})
 * x_i x_j}{x^3} \f}
 */
class KerrHorizonConforming {
 public:
  KerrHorizonConforming() = default;
  static constexpr size_t dim = 3;
  /*!
   * constructs a Kerr horizon conforming map.
   * \param spin : the dimensionless spin
   */
  explicit KerrHorizonConforming(std::array<double, 3> spin);

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  bool is_identity() const {
    return spin_ == std::array<double, 3>{0., 0., 0.};
  }

  friend bool operator==(const KerrHorizonConforming& lhs,
                         const KerrHorizonConforming& rhs);

  void pup(PUP::er& p);

 private:
  template <typename T>
  void stretch_factor_square(
      const gsl::not_null<tt::remove_cvref_wrap_t<T>*> result,
      const std::array<T, 3>& source_coords) const;

  std::array<double, 3> spin_;
  double spin_mag_sq_;
};
bool operator!=(const KerrHorizonConforming& lhs,
                const KerrHorizonConforming& rhs);

}  // namespace domain::CoordinateMaps
