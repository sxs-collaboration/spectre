// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordMapsTimeDependent {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Maps the radius as \f$r(t) = a(t)\rho + \left(b(t) - a(t)\right)
 * \frac{\rho^3} {R^2}\f$ where \f$\rho\f$ is the radius of the source
 * coordinates.
 *
 * The map scales the radius \f$\rho\f$ in the source coordinates
 * \f$\xi^{\hat{i}}\f$ by a factor \f$a(t)\f$, while the coordinates near the
 * outer boundary \f$R\f$, are scaled by a factor \f$b(t)\f$. Here \f$a(t)\f$
 * and \f$b(t)\f$ are FunctionsOfTime. The target/mapped coordinates are denoted
 * by \f$x^i\f$.
 *
 * The mapped coordinates are given by:
 *
 * \f{align}{
 * x^i = \left[a + (b-a) \frac{\rho^2}{R^2}\right] \xi^{\hat{i}}
 *       \delta^i_{\hat{i}},
 * \f}
 *
 * where \f$\xi^{\hat{i}}\f$ are the source coordinates, \f$a\f$ and \f$b\f$ are
 * functions of time, \f$\rho\f$ is the radius in the source coordinates, and
 * \f$R\f$ is the outer boundary.
 *
 * The inverse map is computed by solving the cubic equation:
 *
 * \f{align}{
 * (b-a)\frac{\rho^3}{R^2} + a \rho - r = 0,
 * \f}
 *
 * which is done by defining \f$q=\rho/R\f$, and solving
 *
 * \f{align}{
 * q \left[(b-a) q^2 + a\right] - \frac{r}{R} = 0.
 * \f}
 *
 * The source coordinates are obtained using:
 *
 * \f{align}{
 * \xi^{\hat{i}} = \frac{qR}{r} x^i(t) \delta^{\hat{i}}_i
 * \f}
 *
 * The Jacobian is given by:
 *
 * \f{align}{
 * \frac{\partial x^i}{\partial \xi^{\hat{i}}}=
 *    \left[a + (b-a) \frac{\rho^2}{R^2}\right] \delta^i_{\hat{i}}
 *    + \frac{2 (b-a)}{R^2} \xi^{\hat{j}} \delta^i_{\hat{j}} \xi^{\hat{k}}
 *      \delta_{\hat{k}\hat{i}}
 * \f}
 *
 * The inverse Jacobian is given by:
 *
 * \f{align}{
 * \frac{\partial \xi^{\hat{i}}}{\partial x^i}=
 * \frac{1}{\left[a + (b-a)\rho^2/R^2\right]}
 *  \left[\delta^{\hat{i}}_i -
 *        \frac{2 (b-a)}{\left[a R^2 + 3(b-a)\rho^2\right]}
 *        \xi^{\hat{i}}\xi^{\hat{j}}\delta_{\hat{j}i}\right]
 * \f}
 *
 * The mesh velocity \f$v_g^i\f$ is given by:
 *
 * \f{align}{
 * v_g^i = \left[\frac{da}{dt} + \left(\frac{db}{dt}-\frac{da}{dt}\right)
 *         \frac{\rho^2}{R^2}\right] \xi^{\hat{i}} \delta^i_{\hat{i}}.
 * \f}
 */
template <size_t Dim>
class CubicScale {
 public:
  static constexpr size_t dim = Dim;

  explicit CubicScale(double outer_boundary,
                      std::string function_of_time_name_a,
                      std::string function_of_time_name_b) noexcept;
  CubicScale() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  /// Returns boost::none if the point is outside the range of the map.
  template <typename T>
  boost::optional<std::array<tt::remove_cvref_wrap_t<T>, Dim>> inverse(
      const std::array<T, Dim>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> frame_velocity(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const CubicScale<LocalDim>& lhs,
                         const CubicScale<LocalDim>& rhs) noexcept;

  std::string f_of_t_a_{};
  std::string f_of_t_b_{};
  double one_over_outer_boundary_{std::numeric_limits<double>::signaling_NaN()};
  bool functions_of_time_equal_{false};
};

template <size_t Dim>
bool operator!=(const CubicScale<Dim>& lhs,
                const CubicScale<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordMapsTimeDependent
}  // namespace domain
