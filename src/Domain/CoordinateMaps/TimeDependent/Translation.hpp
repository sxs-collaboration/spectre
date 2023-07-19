// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

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
 * \brief Translation map defined by \f$\vec{x} = \vec{\xi}+F(r)\vec{T}(t)\f$.
 *
 * \details The map adds a translation, \f$F(r)\vec{T}(t)\f$, to the coordinates
 * \f$\vec{\xi}\f$, where \f$\vec{T}(t)\f$ is a FunctionOfTime and\f$F(r)\f$ is
 * a 1D radial MathFunction. The radius of each point is found by subtracting
 * the center map argument from the coordinates \f$\vec{\xi}\f$ or the target
 * coordinates \f$\vec{\bar{\xi}}\f$. The Translation Map class is overloaded so
 * that the user can choose if a radial depedence is needed for their problem.
 * If a radial dependence is not specified, this sets \f$F(r) = 1\f$.
 *
 * ### Mapped Coordinates
 *
 * The translation map translates the coordinates \f$\vec{\xi}\f$ to the
 * target coordinates
 * \f{equation}{
 * \vec{\bar{\xi}} = \vec{\xi} + F(r)\vec{T}(t)
 * \f}
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
 * The inverse translation translates the coordinate \f$\vec{\bar{\xi}}\f$ to
 * the original coordinates
 * \f{equation}{
 * \vec{\xi} = \vec{\bar{\xi}} - F(R)\vec{T}(t)
 * \f}
 * Where R is the root where \f$F(r)\f$ and \f$\vec{T}(t)\f$
 * cross zero. This is done by bracketing and finding the root of
 * \f{equation}{
 * \bar{r}^2 + F(x) (A + |\vec{T}(t)| F(x)) - x^2
 * \f}
 * This equation comes from the SpEC implementation where \f$x\f$ is the root
 * guess and \f$A\f$ is the Root Factor given by \f{equation}{ A = -2.0
 * (\vec{\xi} - \vec{c}) \cdot \vec{T}(t) \f}
 *
 * ### Frame Velocity
 *
 * The frame velocity is given by
 * \f{equation}{
 * \vec{v} = \frac{\vec{dT}(t)}{dt} F(r)
 * \f}
 * where \f$\frac{\vec{dT}(t)}{dt}\f$ is the first derivative of the
 * FunctionOfTime.
 *
 * ### Jacobian
 * The jacobian is computed through either first or second derivatives of the
 * radial MathFunction based on your radius. When the radius is bigger
 * than 1.e-13, the jacobian is given by:
 * \f{equation}{
 * {J^{i}}_{j} = \frac{dF(r)}{dr} T(t)^i \frac{(\xi_j - c_j)}{r}
 * \f}
 * Where \f$\frac{dF(r)}{dr}\f$ is the first derivative of the MathFunction,
 * \f$\vec{\xi_j}\f$ is the source coordinates, \f$\vec{c}\f$ is the center of
 * your map, and r is the radius.
 *
 * At a radius smaller than 1e-13, we use L'Hopital's rule to compute the
 * jacobian as
 * \f{equation}{
 * {J^{i}}_{j} = \frac{d^2F(r)}{dr^2} T(t)^i (\xi_j - c_j)
 * \f}
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

 private:
  template <size_t LocalDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const Translation<LocalDim>& lhs, const Translation<LocalDim>& rhs);

  // This helper function computes the translated coordinates or frame velocity
  // based on the option passed in, 0 for translated coordinates, and frame
  // velocity for any other number.
  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> coord_helper(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      size_t function_or_deriv_index) const;

  double root_finder(double target_radius, double squared_function_of_time,
                     double root_factor) const;

  std::string f_of_t_name_{};
  std::unique_ptr<MathFunction<1, Frame::Inertial>> f_of_r_{};
  std::array<double, Dim> center_{};
};

template <size_t Dim>
inline bool operator!=(const Translation<Dim>& lhs,
                       const Translation<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace domain::CoordinateMaps::TimeDependent
