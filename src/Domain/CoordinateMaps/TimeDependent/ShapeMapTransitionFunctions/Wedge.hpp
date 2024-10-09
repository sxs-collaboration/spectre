// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"

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
 * where, in general,
 *
 * \begin{equation}
 * D(r, \theta, \phi) = R\left(\frac{1 -
 * s}{\sqrt{3}\max(|\sin{\theta}\cos{\phi}|,|\sin{\theta}\sin{\phi}|,
 * |\cos{\theta}|)} + s\right).
 * \label{eq:distance}
 * \end{equation}
 *
 * Here, $s$ is the sphericity of the surface which goes from 0 (flat) to 1
 * (spherical), $R$ is the radius of the spherical surface, $\text{out}$ is the
 * outer surface, and $\text{in}$ is the inner surface. If the sphericity is 1,
 * then the surface is a sphere so $D = R$. If the sphericity is 0, then the
 * surface is a cube. This cube is circumscribed by a sphere of radius $R$. See
 * `domain::CoordinateMaps::Wedge` for more of an explanation of these boundary
 * surfaces and their sphericities.
 *
 * \note Because the shape map distorts only radii and does not affect angles,
 * $D$ is not a function of $r$ so we have that $D(r,\theta,\phi) =
 * D(\theta, \phi)$.
 *
 * There are several assumptions made for this mapping:
 *
 * - The coordinates $r, \theta, \phi$ are assumed to be from the center of the
 *   inner surface, not the center of the computational domain.
 * - The wedges are concentric. (see the constructor) This is also enforced by
 *   the center of the inner surface being within the outer surface
 * - If the centers of the inner and outer surface are different, then the inner
 *   sphericity must be 1 (i.e. $D_{\text{in}} = R$)
 * - The $\max$ in the denominator of $\ref{eq:distance}$ can be simplified a
 *   bit to $\max(|x|, |y|, |z|)/r$. It was written the other way in
 *   $\ref{eq:distance}$ to emphasize that $D$ has no radial dependence.
 *
 * ### Vector formulation
 *
 * \image html WedgeTransition.jpg "Useful vectors for Wedge transition"
 *
 * It can be more convenient to work with vectors rather than just distances for
 * the following equations. Therefore, we define the projection center $\vec P$
 * which points from the center of the outer surface to the center of the inner
 * surface. If the centers of the inner and outer surface are the same, then
 * $\vec P = 0$. Then, we define $\vec x$ as the vector from the center of the
 * outer surface to the coordinate we are interested in. We define $\vec x_0$ as
 * the vector from the center of the outer surface to the point along ray $\vec
 * x - \vec P$ which intersects the inner surface. Similarly, $\vec x_1$
 * intersects the outer surface along the same ray as $\vec x - \vec P$. Any
 * cube has a side length of $2L$.
 *
 * In this way, we can reformulate Eq. $\ref{eq:transition_func}$ as
 *
 * \begin{equation}\label{eq:transition_func_vec}
 *     f(\vec x) = \frac{|\vec x_1 - \vec P| - r}{|\vec x_1 - \vec P| - |\vec
 * x_0 - \vec P|}
 * \end{equation}
 *
 * where
 *
 * \begin{equation}
 *     r  = |\vec x - P|
 * \end{equation}
 *
 * #### Computing vector to inner surface
 *
 * Similar to Eq. $\ref{eq:distance}$ we can define $\vec x_0$ as
 *
 * \begin{equation}\label{eq:x_0_vector}
 *     \vec x_0 = R_0\left( \frac{r(1-s_0)}{\sqrt{3}\max{(x_i - P_i)}} +
 * s\right) \hat r + \vec P
 * \end{equation}
 *
 * with
 *
 * \begin{equation}
 *     \hat r = \frac{\vec x - \vec P}{|\vec x - \vec P|}.
 * \end{equation}
 *
 * and also
 *
 * \begin{equation}\label{eq:x_0_norm}
 *     |\vec x_0 - \vec P| = R_0\left( \frac{r(1-s_0)}{\sqrt{3}\max{(x_i -
 * P_i)}} + s_0\right)
 * \end{equation}
 *
 * ### Computing vector to outer surface
 *
 * Then, we define
 *
 * \begin{equation}\label{eq:x_1}
 *     \vec x_1 = \lambda (\vec x - \vec P) + \vec P
 * \end{equation}
 *
 * where $\lambda$ is a scalar to be determined. We can see that this is valid
 * because $\vec x - \vec P$ is centered on $\vec P$ and lies along the ray so
 * $\lambda$ will scale this vector to be the correct length.
 *
 * Computing $\lambda$ is a bit complicated because the outer surface can either
 * be a cube, a sphere, or something in between and the way to solve for
 * $\lambda$ in the cube case is different than in the sphere case. Therefore we
 * define $\lambda_c$ and $\lambda_s$ as the scalar factors for the cube and
 * sphere case, respectively. This allows us to define $\lambda$ as
 *
 * \begin{equation}\label{eq:lambda}
 *     \lambda = (1 - s_1) \lambda_c + s_1 \lambda_s
 * \end{equation}
 *
 * #### Computing cube lambda
 *
 * To compute $\lambda_c$ for the cube case, we notice that (in the image)
 *
 * \begin{align}
 *     L &= \vec x_1 \cdot \hat z \\
 *     &= \left(\lambda_c^{+z}(\vec x - \vec P) + \vec P\right) \cdot \hat z\\
 *     &= \lambda_c^{+z}(x_z - P_z) + P_z
 * \end{align}
 *
 * If we generalize this to all six unit vectors and solve for $\lambda_c^{\pm
 * k}$ we get
 *
 * \begin{equation}\label{eq:lambda_c_pm}
 *     \lambda_c^{\pm k} = \frac{\pm L - \vec P \cdot \hat x_k}{(\vec x -
 * \vec P) \cdot \hat x_k}
 * \end{equation}
 *
 * for $k \in \{x, y, z\}$. This quantity $\lambda_c^{\pm k}$ represents the
 * factor needed to scale the vector $\vec x - \vec P$ to intersect with the
 * infinite planes that contain the six faces of the cube.
 *
 * Some $\lambda_c^{\pm k}$ will be negative as they point to the opposite
 * planes. We can discard those. Then for the $\lambda_c^{\pm k}$ that are
 * positive, we want the one that is smallest in magnitude. This can be written
 * as
 *
 * \begin{equation}\label{eq:lambda_c}
 *     \lambda_c = \min_{\substack{\lambda_c^{\pm k} > 0}} (\lambda_c^{\pm k})
 * \end{equation}
 *
 * #### Computing sphere lambda
 *
 * To compute $\lambda_s$ for the sphere case, again we notice that (in the
 * image)
 *
 * \begin{align}
 *     R_1^2 &= |\vec x_1|^2 \\
 *     &= |\lambda_s (\vec x - \vec P) + \vec P|^2 \\
 *     &= \lambda_s^2 |\vec x - \vec P|^2 + 2 \lambda_s (\vec x - \vec P) \cdot
 * \vec P + |\vec P|^2
 * \end{align}
 *
 * This is simply a quadratic equation for $\lambda_s$:
 *
 * \begin{equation}\label{eq:lambda_s}
 *     0 = \lambda_s^2 |\vec x - \vec P|^2 + 2 \lambda_s (\vec x - \vec P) \cdot
 * \vec P + |\vec P|^2 - R_1^2
 * \end{equation}
 *
 * We explicitly don't write the typical form of the solution of a quadratic
 * equation because that form behaviors poorly numerically if the two terms in
 * the numerator are close to each other (bad numerical roundoff). Therefore, we
 * just resolve to compute the solution numerically, and say that we have
 * $\lambda_s$.
 *
 * We have now solved for $\vec x_1$.
 *
 * \begin{equation}
 *     \vec x_1 = \left((1 - s_1) \lambda_c + s_1 \lambda_s \right) (\vec x -
 * \vec P) + \vec P
 * \end{equation}
 *
 * given the solutions for $\lambda_c$ in Eqn. $\ref{eq:lambda_c}$ and
 * $\lambda_s$ in Eqn. $\ref{eq:lambda_s}$.
 *
 * ## Original radius divided by mapped radius
 *
 * Given an already mapped point $\tilde{\vec x}$ and the distorted radius
 * $\Sigma(\theta, \phi) = \sum_{\ell,m}\lambda_{\ell m}Y_{\ell m}(\theta,
 * \phi)$, we can figure out the ratio of the original radius $r$ to the mapped
 * $\tilde{r} = |\tilde{\vec x}|$ by solving
 *
 * \begin{equation}
 * \frac{r}{\tilde{r}} =
 * \frac{1}{1-\frac{f(r,\theta,\phi)}{r}\Sigma(\theta,\phi)}
 * \end{equation}
 *
 * After plugging in the transition and solving, we get
 *
 * \begin{equation}
 * \frac{r}{\tilde{r}} = \frac{1 + \frac{|\vec x_1 - \vec P|\Sigma(\theta,
 * \phi)}{\tilde{r}(|\vec x_1 - \vec P| - |\vec x_0 - \vec P|)}}{1 +
 * \frac{\Sigma(\theta, \phi)}{|\vec x_1 - \vec P| - |\vec x_0 - \vec P|}}
 * \label{eq:r_over_rtil}
 * \end{equation}
 *
 * \note Since $\vec x_0 - \vec P$ and $\vec x_1 - \vec P$ are not functions of
 * the radius, we can use $\tilde{\vec x}$ in Eq. $\ref{eq:x_0_vector}$ instead
 * of $\vec x$.
 *
 * ## Gradient
 *
 * The cartesian gradient of the transition function is
 *
 * \begin{equation}
 * \frac{\partial f}{\partial x_i} = \frac{\frac{\partial
 * |\vec x_1 - \vec P|}{\partial x_i} - \frac{x_i - P_i}{r}}{|\vec x_1 - \vec P|
 * - |\vec x_0 - \vec P|} - \frac{\left(|\vec x_1 - \vec P| -
 * r\right)\left(\frac{\partial |\vec x_1 - \vec P|}{\partial x_i} -
 * \frac{\partial |\vec x_0 - \vec P|}{\partial x_i} \right)}{\left(|\vec x_1 -
 * \vec P| - |\vec x_0 - \vec P|\right)^2}.
 * \end{equation}
 *
 * Therefore, we need to compute the gradients of $\vec x_0$ and $\vec x_1$.
 *
 * ### Gradient of vector to inner surface
 *
 * If $\vec P \neq 0$, then we require the inner surface to be a sphere, which
 * means the gradient is trivially zero. Therefore the following is only for
 * when $\vec P = 0$.
 *
 * The gradient of $|\vec x_0 - \vec P|$ depends on the result of the $\max$ in
 * the denominator of $\ref{eq:x_0_norm}$. To simplify the expression, let $i
 * \in \{0,1,2\}$ correspond to the $\max(x_i - P_i)$. Then we can write the
 * gradient of as
 *
 * \begin{align}
 * \frac{\partial |\vec x_0 - \vec P|}{\partial x_i} &= -\text{sgn}(x_i -
 * P_i)\frac{R(1-s)\left(r^2 - (x_i - P_i)^2\right)}
 * {r (x_i - P_i)^2 \sqrt{3}} \\
 * \frac{\partial |\vec x_0 - \vec P|}{\partial x_{i+1}} &=
 * \frac{R(1-s)(x_{i+1} - P_{i+1})}{r |x_i - P_i| \sqrt{3}} \\
 * \frac{\partial |\vec x_0 - \vec P|}{\partial x_{i+2}} &=
 * \frac{R(1-s)(x_{i+2} - P_{i+2})}{r |x_i - P_i| \sqrt{3}}
 * \end{align}
 *
 * where $i+1$ and $i+2$ are understood to be $\mod 3$. Also since we don't
 * allow points at the center of the inner surfaces, we can be assured that for
 * whichever $i$ is max, that $x_i - P_i$ won't be zero.
 *
 * ### Gradient of vector to outer surface
 *
 * This gradient is much more complicated because of how we have to define
 * $\lambda$. To start off with, we need to compute
 *
 * \begin{align}\label{eq:x_1_grad}
 *     \frac{\partial}{\partial x_i}|\vec x_1 - \vec P| &=
 * \frac{\partial}{\partial x_i} (\lambda |\vec x - \vec P|) \\
 *     &= \frac{\lambda (x_i - P_i)}{|\vec x - \vec P|} + \frac{\partial
 * \lambda}{\partial x_i}|\vec x - \vec P|
 * \end{align}
 *
 * Since we know $\lambda$ and $\vec x$, that just leaves
 *
 * \begin{equation}
 *     \frac{\partial\lambda}{\partial x_i} = \frac{\partial\lambda_c}{\partial
 * x_i}(1-s_1) + s_1\frac{\partial\lambda_s}{\partial x_i}
 * \end{equation}
 *
 * #### Gradient of cube lambda
 *
 * Taking the gradient of Eqn. $\ref{eq:lambda_c_pm}$ we notice that $L$, $\vec
 * P$, and $\hat x_k$ are constant, so we only have to worry about $\vec x$
 *
 * \begin{equation}\label{eq:tmp_lambda_c_grad}
 *     \frac{\partial}{\partial x_i}\lambda_c^{\pm k} = -\frac{(\pm L -
 * \vec P \cdot \hat x_k)(\partial/\partial x_i(\vec x - \vec P) \cdot \hat
 * x_k)}{((\vec x - \vec P) \cdot \hat x_k)^2}
 * \end{equation}
 *
 * The gradient in the numerator is simple to take so we end up with
 *
 * \begin{equation}
 *     \frac{\partial}{\partial x_i}\lambda_c^{\pm k} = -\frac{(\pm L -
 * \vec P \cdot \hat x_k)}{( x_k - P_k)^2} \delta_{ik}
 * \end{equation}
 *
 * #### Gradient of sphere lambda
 *
 * Taking the gradient of Eqn. $\ref{eq:lambda_s}$ (and using the fact that
 * $\vec P$ and $R$ are constant), we get
 *
 * \begin{equation}
 *     0 = \lambda_s \frac{\partial \lambda_s}{\partial x_i}|\vec x - \vec P|^2
 * + \lambda_s^2 (\vec x - \vec P) \cdot \frac{\partial}{\partial x_i}(\vec x -
 * \vec P) + \frac{\partial\lambda_s}{\partial x_i}(\vec x - \vec P) \cdot \vec
 * P + \lambda_s \vec P \cdot \frac{\partial}{\partial x_i}(\vec x - \vec P)
 * \end{equation}
 *
 * Taking the simple gradients and moving things around, we get
 *
 * \begin{equation}
 *     \frac{\partial\lambda_s}{\partial x_i} = - \frac{\lambda_s^2 (x_i - P_i)
 * + \lambda_s P_i}{\lambda_s |\vec x - \vec P|^2 + (\vec x - \vec P) \cdot \vec
 * P}
 * \end{equation}
 *
 * where $\lambda_s$ can be computed from Eqn. $\ref{eq:lambda_s}$.
 *
 * So now we can put it all together and we get
 *
 * \begin{equation}
 *    \frac{\partial\lambda}{\partial x_i} =  -\frac{(\pm L - \vec P \cdot
 * \hat x_k)}{( x_k - P_k)^2} \delta_{ik}(1 - s_1)
 *     + s_1 \frac{\lambda_s^2 (x_i - P_i) + \lambda_s P_i}{\lambda_s |\vec x -
 * \vec P|^2 + (\vec x - \vec P) \cdot \vec P} \end{equation}
 *
 * which can then be substituted into Eqn. $\ref{eq:x_1_grad}$ to get the
 * gradient of $|\vec x_1 - \vec P|$.
 */
class Wedge final : public ShapeMapTransitionFunction {
  struct Surface {
    std::array<double, 3> center{};
    double radius{};
    double sphericity{};
    double half_cube_length{};

    Surface() = default;
    Surface(const std::array<double, 3>& center_in, double radius_in,
            double sphericity_in);

    void pup(PUP::er& p);
  };

 public:
  /*!
   * \brief Class to represent the direction of the wedge relative to the outer
   * center.
   */
  enum class Axis : int {
    PlusZ = 3,
    MinusZ = -3,
    PlusY = 2,
    MinusY = -2,
    PlusX = 1,
    MinusX = -1,
    None = 0
  };

  friend std::ostream& operator<<(std::ostream& os, Axis axis);

 private:
  static size_t axis_index(Axis axis);
  static double axis_sgn(Axis axis);

 public:
  explicit Wedge() = default;

  /*!
   * \brief Construct a Wedge transition for a wedge block in a given direction.
   *
   * \details Many concentric wedges can be part of the same falloff from 1 at
   * the inner boundary of the innermost wedge, to 0 at the outer boundary of
   * the outermost wedge.
   *
   * \note If \p inner_center and \p outer_center are different, then
   * \p inner_sphericity must be 1.0.
   *
   * \param inner_center Center of the inner surface
   * \param inner_radius Inner radius of innermost wedge
   * \param inner_sphericity Sphericity of innermost surface of innermost wedge
   * \param outer_center Center of the outer surface
   * \param outer_radius Outermost radius of outermost wedge
   * \param outer_sphericity Sphericity of outermost surface of outermost wedge
   * \param axis The direction that this wedge is in.
   * \param reverse If true, the transition function will be 0 at the inner
   * boundary and 1 at the outer boundary (useful for deforming star surfaces).
   * Otherwise, it will be 1 at the inner boundary and 0 at the outer boundary
   * (useful for deforming black hole excision surfaces).
   */
  Wedge(const std::array<double, 3>& inner_center, double inner_radius,
        double inner_sphericity, const std::array<double, 3>& outer_center,
        double outer_radius, double outer_sphericity, Axis axis,
        bool reverse = false);

  double operator()(const std::array<double, 3>& source_coords) const override;
  DataVector operator()(
      const std::array<DataVector, 3>& source_coords) const override;

  std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const override;

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
  std::array<T, 3> gradient_impl(const std::array<T, 3>& source_coords) const;

  // This is x_0 - P in the docs
  template <typename T>
  std::array<T, 3> compute_inner_surface_vector(
      const std::array<T, 3>& centered_coords,
      const T& centered_coords_magnitude,
      const std::optional<Axis>& potential_axis) const;

  // This is x_1 - P in the docs
  template <typename T>
  std::array<T, 3> compute_outer_surface_vector(
      const std::array<T, 3>& centered_coords, const T& lambda) const;

  template <typename T>
  T lambda_cube(const std::array<T, 3>& centered_coords,
                const std::optional<Axis>& potential_axis) const;

  template <typename T>
  T lambda_sphere(const std::array<T, 3>& centered_coords,
                  const T& centered_coords_magnitude) const;

  template <typename T>
  T compute_lambda(const std::array<T, 3>& centered_coords,
                   const T& centered_coords_magnitude,
                   const std::optional<Axis>& potential_axis) const;

  template <typename T>
  std::array<T, 3> lambda_cube_gradient(
      const T& lambda_cube, const std::array<T, 3>& centered_coords) const;

  template <typename T>
  std::array<T, 3> lambda_sphere_gradient(
      const T& lambda_sphere, const std::array<T, 3>& centered_coords,
      const T& centered_coords_magnitude) const;

  template <typename T>
  std::array<T, 3> compute_lambda_gradient(
      const std::array<T, 3>& centered_coords,
      const T& centered_coords_magnitude) const;

  template <typename T>
  void check_distances(const T& inner_distance, const T& outer_distance,
                       const T& centered_coords_magnitude,
                       const std::array<T, 3>& source_coords) const;

  Surface inner_surface_{};
  Surface outer_surface_{};
  std::array<double, 3> projection_center_{};
  Axis axis_{};
  bool reverse_{false};

  static constexpr double eps_ = std::numeric_limits<double>::epsilon() * 100;
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
