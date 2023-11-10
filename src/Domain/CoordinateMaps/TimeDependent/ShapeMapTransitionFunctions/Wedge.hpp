// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

/*!
 * \brief A transition function that falls off linearly from an inner surface of
 * a wedge to an outer surface of a wedge. Meant to be used in
 * `domain::CoordinateMaps::Wedge` blocks.
 *
 * \details The functional form of this transition is
 *
 * \begin{equation}
 * f(r, \theta, \phi) = \frac{D_{\text{out}}(r, \theta, \phi) -
 * r}{D_{\text{out}}(r, \theta, \phi) - D_{\text{in}}(r, \theta, \phi)},
 * \label{eq:transition_func}
 * \end{equation}
 *
 * where
 *
 * \begin{equation}
 * D(r, \theta, \phi) = R\left(\frac{1 - s}{\sqrt{3}\cos{\theta}} + s\right),
 * \end{equation}
 *
 * and $s$ is the sphericity of the surface which goes from 0 (flat) to 1
 * (spherical), $R$ is the radius of the spherical surface, $\text{out}$ is the
 * outer surface, and $\text{in}$ is the inner surface. If the sphericity is 1,
 * then the surface is a sphere so $D = R$. If the sphericity is 0, then the
 * surface is a cube. This cube is circumscribed by a sphere of radius $R$. See
 * `domain::CoordinateMaps::Wedge` for more of an explanation of these boundary
 * surfaces and their sphericities.
 *
 * \note Because the shape map distorts only radii and does not affect angles,
 * $D$ is only a function of $\theta$ so we have that $D(r,\theta,\phi) =
 * D(\theta)$.
 *
 * There are several assumptions made for this mapping:
 *
 * - The `domain::CoordinateMaps::Wedge`s have an opening angle of
 *   $\frac{\pi}{2}$ in each direction.
 * - The coordinates $r, \theta, \phi$ are assumed to be from the center of the
 *   wedge, not the center of the computational domain.
 * - The wedges are concentric. (see the constructor)
 * - Eq. $\ref{eq:transition_func}$ assumes it is in a wedge that is aligned
 *   with the +z direction. Therefore, $\cos{\theta} = \hat{r} \cdot \hat{z} =
 *   \frac{\vec{r}}{r} \cdot \hat{z} = \frac{z}{r}$. For the wedges aligned with
 *   the other axes, the coordinates are first rotated so they align with +z,
 *   then the computation is done.
 *
 * ## Gradient
 *
 * The cartesian gradient of the transition function is
 *
 * \begin{equation}
 * \frac{\partial f}{\partial x_i} = \frac{\frac{\partial
 * D_{\text{out}}}{\partial x_i} - \frac{x_i}{r}}{D_{\text{out}} -
 * D_{\text{in}}} - \frac{\left(D_{\text{out}} - r\right)\left(\frac{\partial
 * D_{\text{out}}}{\partial x_i} - \frac{\partial
 * D_{\text{in}}}{\partial x_i} \right)}{\left(D_{\text{out}} -
 * D_{\text{in}}\right)^2}
 * \end{equation}
 *
 * where
 *
 * \f{align}{
 * \frac{\partial D}{\partial z} &= \frac{R\left(1-s\right)}{\sqrt{3}}
 * \frac{\partial}{\partial z} \left(\frac{z}{r} \right) =
 * \frac{R\left(1-s\right)}{\sqrt{3}} \left(\frac{r^2 - z^2}{r^3} \right), \\
 * \frac{\partial D}{\partial x} &= \frac{R\left(1-s\right)}{\sqrt{3}}
 * \frac{\partial}{\partial x} \left(\frac{z}{r} \right) =
 * -\frac{R\left(1-s\right)}{\sqrt{3}} \left(\frac{xz}{r^3} \right), \\
 * \frac{\partial D}{\partial y} &= \frac{\partial D}{\partial x} \left(x
 * \rightarrow y \right),
 * \f}
 *
 * making use of the fact that this wedge is aligned with the +z axis so that
 * $\cos{\theta} = \frac{z}{r}$.
 *
 * ## Original radius divided by mapped radius
 *
 * Given an already mapped point and the distorted radius $\Sigma(\theta, \phi)
 * = \sum_{\ell,m}\lambda_{\ell,m}Y_{\ell,m}(\theta, \phi)$, we can figure out
 * the ratio of the original radius $r$ to the mapped $\tilde{r}$ by solving
 *
 * \begin{equation}
 * \frac{r}{\tilde{r}} = \frac{1}{1-f(r,\theta,\phi)\Sigma(\theta,\phi)}
 * \end{equation}
 *
 * Because this is a linear transition, this will result in is having to solve a
 * quadratic equation of the form
 *
 * \begin{equation}
 * \tilde{r} x^2 + \left(\frac{D_{\text{out}} -
 * D_{\text{in}}}{\Sigma(\theta,\phi)} - D_{\text{out}}\right) x -
 * \frac{D_{\text{out}} - D_{\text{in}}}{\Sigma(\theta,\phi)} = 0
 * \label{eq:r_over_rtil}
 * \end{equation}
 *
 * where $x = \frac{r}{\tilde{r}}$.
 *
 * \note Since $D$ is not a function of the radius, we can treat it as a
 * constant in Eq. $\ref{eq:r_over_rtil}$ and compute the angles using
 * $\tilde{r}$.
 *
 * ## Special cases
 *
 * The equations above become simpler if the inner boundary, outer boundary, or
 * both are spherical ($s = 1$). If a boundary is spherical, then $D = R$ and
 * $\frac{\partial D}{\partial x_i} = 0$. Since $D$ can be held constant when
 * solving $\ref{eq:r_over_rtil}$, the quadratic equation itself doesn't
 * simplify, only the computation of $D$.
 *
 * If the inner boundary is spherical (but not the outer boundary), then the
 * gradient of $f$ becomes
 *
 * \begin{equation}
 * \frac{\partial f}{\partial x_i} = \frac{\frac{\partial
 * D_{\text{out}}}{\partial x_i} - \frac{x_i}{r}}{D_{\text{out}} -
 * D_{\text{in}}} - \frac{\left(D_{\text{out}} - r\right) - \left(\frac{\partial
 * D_{\text{out}}}{\partial x_i} \right)}{\left(D_{\text{out}} -
 * D_{\text{in}}\right)^2}.
 * \end{equation}
 *
 * If the outer boundary is spherical (but not the inner boundary), then the
 * gradient of $f$ becomes
 *
 * \begin{equation}
 * \frac{\partial f}{\partial x_i} = \frac{-\frac{x_i}{r}}{D_{\text{out}} -
 * D_{\text{in}}} - \frac{\left(D_{\text{out}} - r\right) + \left(\frac{\partial
 * D_{\text{in}}}{\partial x_i} \right)}{\left(D_{\text{out}} -
 * D_{\text{in}}\right)^2}
 * \end{equation}
 *
 * If the both boundaries are spherical, then the gradient of $f$ simplifies
 * greatly to
 *
 * \begin{equation}
 * \frac{\partial f}{\partial x_i} = \frac{-\frac{x_i}{r}}{D_{\text{out}} -
 * D_{\text{in}}}
 * \end{equation}
 */
class Wedge final : public ShapeMapTransitionFunction {
  struct Surface {
    double radius{};
    double sphericity{};

    // This is the distance from the center (assumed to be 0,0,0) to this
    // surface in the same direction as coords.
    template <typename T>
    T distance(const std::array<T, 3>& coords) const;

    void pup(PUP::er& p);

    bool operator==(const Surface& other) const;
    bool operator!=(const Surface& other) const;
  };

 public:
  explicit Wedge() = default;

  /*!
   * \brief Construct a Wedge transition for a wedge block in a given direction.
   *
   * \details Many concentric wedges can be part of the same falloff from 1 at
   * the inner boundary of the innermost wedge, to 0 at the outer boundary of
   * the outermost wedge.
   *
   * \param inner_radius Inner radius of innermost wedge
   * \param outer_radius Outermost radius of outermost wedge
   * \param inner_sphericity Sphericity of innermost surface of innermost wedge
   * \param outer_sphericity Sphericity of outermost surface of outermost wedge
   * \param orientation_map The same orientation map that was passed into the
   * corresponding `domain::CoordinateMaps::Wedge` block.
   */
  Wedge(double inner_radius, double outer_radius, double inner_sphericity,
        double outer_sphericity, OrientationMap<3> orientation_map);

  double operator()(const std::array<double, 3>& source_coords) const override;
  DataVector operator()(
      const std::array<DataVector, 3>& source_coords) const override;

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const override;

  double map_over_radius(
      const std::array<double, 3>& source_coords) const override;
  DataVector map_over_radius(
      const std::array<DataVector, 3>& source_coords) const override;

  std::array<double, 3> gradient(
      const std::array<double, 3>& source_coords) const override;
  std::array<DataVector, 3> gradient(
      const std::array<DataVector, 3>& source_coords) const override;

  WRAPPED_PUPable_decl_template(Wedge);
  explicit Wedge(CkMigrateMessage* msg);
  void pup(PUP::er& p) override;

  std::unique_ptr<ShapeMapTransitionFunction> get_clone() const override {
    return std::make_unique<Wedge>(*this);
  }

  bool operator==(const ShapeMapTransitionFunction& other) const override;
  bool operator!=(const ShapeMapTransitionFunction& other) const override;

 private:
  template <typename T>
  T call_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  T map_over_radius_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  template <typename T>
  void check_distances(const std::array<T, 3>& coords) const;

  Surface inner_surface_{};
  Surface outer_surface_{};
  OrientationMap<3> orientation_map_{};
  Direction<3> direction_{};
  static constexpr double eps_ = std::numeric_limits<double>::epsilon() * 100;
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
