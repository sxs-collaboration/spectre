// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain::CoordinateMaps::TimeDependent {

/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Distorts a distribution of points radially according to a spherical
 * harmonic expansion while preserving angles.
 *
 * \details The shape map distorts the distance \f$r\f$ between a point and
 * the center while leaving the angles \f$\theta\f$, \f$\phi\f$ between them
 * preserved by applying a spherical harmonic expansion with time-dependent
 * coefficients \f$\lambda_{lm}(t)\f$. There are two ways to specify the
 * time-dependent coefficients \f$\lambda_{lm}(t)\f$:
 *
 * 1. A single FunctionOfTime which specifies all coefficients. This
 *    FunctionOfTime should have `ylm::Spherepack::spectral_size()` number of
 *    components. These are in Spherepack order and should be the Spherepack
 *    coefficients, *not* the spherical harmonic coefficients. See the note
 *    below. To use this, set the `size_function_of_time_name` argument of the
 *    constructor to `std::nullopt`.
 * 2. Two different FunctionOfTime%s. The first is similar to 1.) in that it
 *    should have the same number of components, be in Spherepack order, and be
 *    the Spherepack coefficients. The only difference is that the \f$l = 0\f$
 *    coefficient should be identically 0. The second FunctionOfTime should have
 *    a single component which will be the \f$l = 0\f$ coefficient. This
 *    component should be stored as the spherical harmonic coefficient and *not*
 *    a Spherepack coefficient. See the note below. To use this method, set the
 *    `size_function_of_time_name` argument of the constructor to the name of
 *    the FunctionOfTime that's in the cache. This method is useful if we have
 *    control systems because we have a separate control system controlling a
 *    separate function of time for the \f$l = 0\f$ coefficient than we do for
 *    the other coefficients.
 *
 * \note The quantities stored in the "shape" FunctionOfTime (the
 * `shape_function_of_time_name` argument in the constructor that must always be
 * specified) are ***not*** the complex spherical-harmonic coefficients
 * \f$\lambda_{lm}(t)\f$, but instead are the real-valued SPHEREPACK
 * coefficients \f$a_{lm}(t)\f$ and \f$b_{lm}(t)\f$ used by Spherepack. This
 * is the same for both methods of specifying FunctionOfTime%s above. The
 * relationship between these two sets of coefficients is
 * \f{align}
 * a_{l0} & = \sqrt{\frac{2}{\pi}}\lambda_{l0}&\qquad l\geq 0,\\
 * a_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Re}(\lambda_{lm})
 * &\qquad l\geq 1, m\geq 1, \\
 * b_{lm} & = (-1)^m\sqrt{\frac{2}{\pi}} \mathrm{Im}(\lambda_{lm})
 * &\qquad l\geq 1, m\geq 1.
 * \f}
 * The "shape" FunctionOfTime stores coefficients only for non-negative \f$m\f$;
 * this is because the function we are expanding is real, so the
 * coefficients for \f$m<0\f$ can be obtained from \f$m>0\f$ coefficients by
 * complex conjugation.
 * If the `size_function_of_time_name` argument is given to the constructor,
 * then it is asserted that the \f$l=0\f$ coefficient of the "shape" function of
 * time is exactly 0. The \f$l=0\f$ coefficient is then controlled by the "size"
 * FunctionOfTime. Unlike the "shape" FunctionOfTime, the quantity in the
 * "size" FunctionOfTime ***is*** the "complex" spherical harmonic coefficient
 * \f$\lambda_{00}(t)\f$, and not the SPHEREPACK coefficient \f$a_{00}(t)\f$
 * ("complex" is in quotes because all \f$m=0\f$ coefficients are always real.)
 * Here and below we write the equations in terms of \f$\lambda_{lm}(t)\f$
 * instead of \f$a_{lm}(t)\f$ and \f$b_{lm}(t)\f$, regardless of which
 * FunctionOfTime representation we are using, because the resulting expressions
 * are much shorter.
 *
 * \parblock
 *
 * \note Also note that the FunctionOfTime coefficients $\lambda_{lm}(t)$ are
 * stored as *negative* of the coefficients you'd retrieve from a
 * `Strahlkorper`. This is because you would typically represent the expansion
 * of a strahlkorper as $S(r) = +\sum S_{lm} Y_{lm}$. However, in equation
 * $\ref{eq:map_form_2}$ there is a minus sign on the $\sum \lambda_{lm}
 * Y_{lm}$, not a plus sign. Therefore, $\lambda_{lm}(t)$ picks up an extra
 * factor of $-1$. This is purely a choice of convention.
 *
 * \endparblock
 *
 * An additional dimensionless domain-dependent transition function
 * \f$f(r, \theta, \phi)\f$ ensures that the distortion falls off correctly to
 * zero at a certain boundary (must be a block boundary). The function \f$f(r,
 * \theta, \phi)\f$ is restricted such that
 *
 * \f{equation}{
 * 0 \leq f(r, \theta, \phi) \leq 1
 * \f}
 *
 * ### Mapped coordinates
 *
 * Given a point with cartesian coordinates \f$\xi^i\f$, let the polar
 * coordinates \f$(r, \theta, \phi)\f$ with respect to a center \f$x_c^i\f$ be
 * defined in the usual way:
 *
 * \f{align}{
 * \xi^0 - x_c^0 &= r \sin(\theta) \cos(\phi)\\
 * \xi^1 - x_c^1 &= r \sin(\theta) \sin(\phi)\\
 * \xi^2 - x_c^2 &= r \cos(\theta)
 * \f}
 *
 * The shape map maps the unmapped
 * coordinates \f$\xi^i\f$ to coordinates \f$x^i\f$:
 *
 * \f{equation}{\label{eq:map_form_1}
 * x^i = \xi^i - (\xi^i - x_c^i) \frac{f(r, \theta, \phi)}{r} \sum_{lm}
 * \lambda_{lm}(t)Y_{lm}(\theta, \phi).
 * \f}
 *
 * Or written another way
 *
 * \f{equation}{\label{eq:map_form_2}
 * x^i = x_c^i + (\xi^i - x_c^i) \left(1 - \frac{f(r, \theta, \phi)}{r}
 * \sum_{lm} \lambda_{lm}(t)Y_{lm}(\theta, \phi)\right).
 * \f}
 *
 * The form in Eq. \f$\ref{eq:map_form_2}\f$ makes two things
 * clearer
 *
 * 1. This shape map is just a radial distortion about \f$x_c^i\f$
 * 2. The coefficients \f$\lambda_{lm}\f$ have units of distance because
 *    \f$\sum\lambda_{lm}(t)Y_{lm}(\theta,\phi) / r\f$ must be dimensionless
 *    (because \f$f\f$ is dimensionless).
 *
 * ### Inverse map
 *
 * The inverse map is given by:
 * \f{equation}{
 * \xi^i = x_c^i + (x^i-x_c^i)*(r/\tilde{r}),
 * \f}
 * where \f$\tilde{r}\f$ is the radius of \f$\xi\f$, calculated by the
 * transition map. In order to compute \f$r/\tilde{r}\f$, the following equation
 * must be solved
 *
 * \f{equation}{
 * \frac{r}{\tilde{r}} =
 * \frac{1}{1-f(r,\theta,\phi)\sum\lambda_{lm}(t)Y_{lm}(\theta,\phi) / r}
 * \f}
 *
 * For more details, see
 * \link domain::CoordinateMaps::ShapeMapTransitionFunctions::ShapeMapTransitionFunction::original_radius_over_radius
 * ShapeMapTransitionFunction::original_radius_over_radius \endlink.
 *
 * ### Frame velocity
 *
 * The frame velocity \f$v^i\ = dx^i / dt\f$ is calculated trivially:
 * \f{equation}{
 * v^i = - (\xi^i - x_c^i) \frac{f(r, \theta, \phi)}{r} \sum_{lm}
 * \dot{\lambda}_{lm}(t)Y_{lm}(\theta, \phi).
 * \f}
 *
 * ### Jacobian
 *
 * The Jacobian is given by:
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^j} = \delta_j^i &\left( 1 - \frac{f(r,
 * \theta, \phi)}{r} \sum_{lm} \lambda_{lm}(t)Y_{lm}(\theta, \phi)\right)
 * \nonumber \\
 * &- (\xi^i - x_c^i)
 * \left[\left(\frac{1}{r}\frac{\partial}{\partial \xi^j} f(r, \theta, \phi) -
 * \frac{\xi_j}{r^3} f(r, \theta, \phi)\right) \sum_{lm}
 * \lambda_{lm}(t)Y_{lm}(\theta, \phi) + \frac{f(r, \theta, \phi)}{r}
 * \sum_{lm} \lambda_{lm}(t) \frac{\partial}{\partial \xi^j} Y_{lm}(\theta,
 * \phi) \right].
 * \f}
 *
 * where \f$\xi_j = \xi^j\f$.
 *
 * ### Inverse Jacobian
 *
 * The inverse Jacobian is computed by numerically inverting the Jacobian.
 *
 * For future optimization, the `interpolation_info` objects calculated in all
 * functions of this class could be cached. Since every element should evaluate
 * the same grid coordinates most time steps, this might greatly decrease
 * computation. Every element has their own clone of the shape map so the
 * caching could be done with member variables. Care must be taken that
 * `jacobian` currently calculates the `interpolation_info` with an order
 * higher.
 */
class Shape {
 public:
  using FunctionsOfTimeMap = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  explicit Shape(
      const std::array<double, 3>& center, size_t l_max, size_t m_max,
      std::unique_ptr<ShapeMapTransitionFunctions::ShapeMapTransitionFunction>
          transition_func,
      std::string shape_function_of_time_name,
      std::optional<std::string> size_function_of_time_name = std::nullopt);

  Shape() = default;
  ~Shape() = default;
  Shape(Shape&&) = default;
  Shape& operator=(Shape&&) = default;
  Shape(const Shape& rhs);
  Shape& operator=(const Shape& rhs);

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> frame_velocity(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords, double time,
      const FunctionsOfTimeMap& functions_of_time) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
  static bool is_identity() { return false; }
  static constexpr size_t dim = 3;

  const std::unordered_set<std::string>& function_of_time_names() const {
    return f_of_t_names_;
  }

 private:
  std::string shape_f_of_t_name_;
  std::optional<std::string> size_f_of_t_name_;
  std::unordered_set<std::string> f_of_t_names_;
  std::array<double, 3> center_{};
  size_t l_max_ = 2;
  size_t m_max_ = 2;
  ylm::Spherepack ylm_{2, 2};
  std::unique_ptr<ShapeMapTransitionFunctions::ShapeMapTransitionFunction>
      transition_func_;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> center_coordinates(
      const std::array<T, 3>& coords) const {
    return {coords[0] - center_[0], coords[1] - center_[1],
            coords[2] - center_[2]};
  }

  void check_size(const gsl::not_null<DataVector*>& coefs,
                  const FunctionsOfTimeMap& functions_of_time, double time,
                  bool use_deriv) const;

  // Checks that the vector of coefficients has the right size and that the
  // monopole and dipole coefficients are zero.
  void check_coefficients(const DataVector& coefs) const;

  template <typename T>
  T check_and_compute_one_over_radius(
      const std::array<T, 3>& centered_coords) const;

  friend bool operator==(const Shape& lhs, const Shape& rhs);
};
bool operator!=(const Shape& lhs, const Shape& rhs);

}  // namespace domain::CoordinateMaps::TimeDependent
