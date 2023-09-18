// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
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
 * \brief Translation map defined by \f$\vec{x} = \vec{\xi}+F(r)\vec{T}(t)\f$
 * where $F(r)$ takes on different forms based on which constructor is used.
 *
 * \details The map adds a translation to the coordinates $\vec{\xi}$ based on
 * what type of translation is needed. For the piecewise translation, a
 * translation $F(r)\vec{T}(t)$ is added to $\vec{\xi}$ based on what region
 * $|\vec{\xi}|$ is in. For coordinates within the inner radius, $F(r) = 1$
 * causing a uniform translation. Coordinates in between the inner and outer
 * radius have a linear radial falloff applied to them. Coordinates beyond the
 * outer radius have no translation applied to them $F(r) = 0$. The piecewise
 * translation assumes that the center of your map is at (0., 0., 0.). For the
 * radial MathFunction translation, a radial translation \f$F(r)\vec{T}(t)\f$ is
 * added to the coordinates \f$\vec{\xi}\f$, where \f$\vec{T}(t)\f$ is a
 * FunctionOfTime and\f$F(r)\f$ is a 1D radial MathFunction. The radius of each
 * point is found by subtracting the center map argument from the coordinates
 * \f$\vec{\xi}\f$ or the target coordinates \f$\vec{\bar{\xi}}\f$. The
 * Translation Map class is overloaded so that the user can choose between a
 * piecewise translation, radial translation or a uniform translation based on
 * their problem. If a radial dependence is not specified, this sets \f$F(r) =
 * 1\f$.
 *
 * ### Mapped Coordinates
 * The piecewise translation translates the coordinates $\vec{\xi}$
 * to the target coordinates $\vec{\bar{\xi}}$ based on the region $\vec{\xi}$
 * is in.
 * \f{equation}{
 * \vec{\bar{\xi}} = \left\{\begin{array}{ll}\vec{\xi} + \vec{T}(t), &
 * |\vec{\xi}| \leq R_{in}, \\ \vec{\xi} + wT(t), &  R_{in} < |\vec{\xi}| <
 * R_{out}, \\ \vec{\xi}, & |\vec{\xi}| \geq R_{out} \end{array}\right.
 * \f}
 *
 * Where $R_{in}$ is the inner radius, $R_{out}$ is the outer radius, and $w$ is
 * the radial falloff factor found through
 * \f{equation}{
 * w = \frac{R_{out} - |\vec{\xi}|}{R_{out} - R_{in}}
 * \f}
 *
 * The radial MathFunction translation translates the coordinates
 * \f$\vec{\xi}\f$ to the target coordinates \f{equation}{\vec{\bar{\xi}} =
 * \vec{\xi} + F(r)\vec{T}(t) \f}
 *
 * If you only supply a FunctionOfTime to the constructor of this class, the
 * radial function will be set to 1.0 causing a uniform translation for your
 * coordinates. If a FunctionOfTime, MathFunction, and map center are passed in,
 * the radius will be found through
 * \f{equation}{
 * r = |\vec{\xi} - \vec{c}|
 * \f}
 * where r is the radius and \f$\vec{c}\f$ is the center argument.
 *
 * ### Inverse Translation
 * The piecewise inverse translates the coordinates
 * \f$\vec{\bar{\xi}}\f$ to the original coordinates based on what region
 * $\vec{\bar{\xi}}$ is in.
 * \f{equation}{
 * \vec{\xi} = \left\{\begin{array}{ll}\vec{\bar{\xi}} -
 * \vec{T}(t), & |\vec{\bar{\xi}}| \leq R_{in}, or, |\vec{\bar{\xi}} - T(t)|
 * \leq R_{in}, \\
 * \vec{\bar{\xi}} - wT(t), &  R_{in} < |\vec{\bar{\xi}}| < R_{out}, \\
 * \vec{\bar{\xi}}, & |\vec{\bar{\xi}}| \geq R_{out}\end{array}\right.
 * \f}
 * Where $w$ is the radial falloff factor found through a quadratic solve of the
 * form
 * \f{equation}{
 * w^2(\vec{T}(t)^2 - (R_{out} - R_{in})^2) - 2w(\vec{T}(t)\vec{\bar{\xi}} -
 * R_{out}(R_{out} - R_{in})) + \vec{\bar{\xi}}^2 - R_{out}^2
 * \f}
 * The inverse map also assumes that if $\vec{\bar{\xi}}
 * - \vec{T}(t) \leq R_{in}$ then the translated point originally came from
 * within the inner radius so it'll be translated back without a quadratic
 * solve.
 *
 * The radial MathFunction inverse translates the coordinates
 * \f$\vec{\bar{\xi}}\f$ to the original coordinates using
 * \f{equation}{
 * \vec{\xi} = \vec{\bar{\xi}} - F(r)\vec{T}(t)
 * \f}
 * where \f$r^2\f$ is found as the root of
 * \f{equation}{
 *   r^2 = \Big(\vec{\bar{\xi}} - \vec{c} - F(r) \vec{T}(t)\Big)^2.
 * \f}
 *
 * ### Frame Velocity
 * For the piecewise translation, the frame velocity is found through
 * \f{equation}{
 * \vec{v} = \left\{\begin{array}{ll}\frac{\vec{dT}(t)}{dt}, & |\vec{\xi}| \leq
 * R_{in}, \\ w\frac{\vec{dT}(t)}{dt}, &  R_{in} < |\vec{\xi}| < R_{out}, \\ 0,
 * & |\vec{\xi}| \geq R_{out} \end{array}\right.
 * \f}
 *
 * For the radial MathFunction translation, the frame velocity is found through
 * \f{equation}{
 * \vec{v} = \frac{\vec{dT}(t)}{dt} F(r)
 * \f}
 * where \f$\frac{\vec{dT}(t)}{dt}\f$ is the first derivative of the
 * FunctionOfTime.
 *
 * ### Jacobian
 * For the piecewise translation, the jacobian is computed based on what region
 * the coordinates $\vec{\xi}$ is in.
 * \f{equation}{
 * {J^{i}}_{j} = \frac{dw}{dr} T(t)^i \frac{\xi_j}{r}, R_{in} <
 * |\vec{\bar{\xi}}| < R_{out}
 * \f}
 * otherwise, it will return the identity matrix.
 *
 * For the radial MathFunction translation, the jacobian is computed through the
 * first derivative when the radius is bigger than 1.e-13:
 * \f{equation}{
 * {J^{i}}_{j} = \frac{dF(r)}{dr} T(t)^i \frac{(\xi_j - c_j)}{r}
 * \f}
 * Where \f$\frac{dF(r)}{dr}\f$ is the first derivative of the MathFunction,
 * \f$\vec{\xi_j}\f$ is the source coordinates, \f$\vec{c}\f$ is the center of
 * your map, and r is the radius.
 *
 * At a radius smaller than 1e-13, we ASSERT that the radial MathFunction is
 * smooth $\frac{dF(r)}{dr} \approx 0$, so return the identity matrix.
 *
 *
 * ### Inverse Jacobian
 * The inverse jacobian is computed numerically by inverting the jacobian.
 */
template <size_t Dim>
class Translation {
 public:
  static constexpr size_t dim = Dim;

  Translation() = default;
  explicit Translation(std::string function_of_time_name);

  explicit Translation(std::string function_of_time_name, double inner_radius,
                       double outer_radius);

  explicit Translation(
      std::string function_of_time_name,
      std::unique_ptr<MathFunction<1, Frame::Inertial>> radial_function,
      std::array<double, Dim>& center);

  Translation(const Translation<Dim>& Translation_Map);

  ~Translation() = default;
  Translation(Translation&&) = default;
  Translation& operator=(Translation&&) = default;
  Translation& operator=(const Translation& Translation_Map);

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> frame_velocity(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static bool is_identity() { return false; }

  const std::unordered_set<std::string>& function_of_time_names() const {
    return f_of_t_names_;
  }

 private:
  template <size_t LocalDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const Translation<LocalDim>& lhs, const Translation<LocalDim>& rhs);

  // These 2 helper functions compute the translated coordinates or frame
  // velocity based on the option passed in, 0 for translated coordinates, and
  // frame velocity for any other number.
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> math_function_helper(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      size_t function_or_deriv_index) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> piecewise_helper(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      size_t function_or_deriv_index) const;

  double root_finder(const std::array<double, Dim>& distance_to_center,
                     const DataVector& function_of_time) const;

  std::string f_of_t_name_{};
  std::unordered_set<std::string> f_of_t_names_;
  std::optional<double> inner_radius_;
  std::optional<double> outer_radius_;
  std::unique_ptr<MathFunction<1, Frame::Inertial>> f_of_r_{};
  std::array<double, Dim> center_{};
};

template <size_t Dim>
inline bool operator!=(const Translation<Dim>& lhs,
                       const Translation<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace domain::CoordinateMaps::TimeDependent
