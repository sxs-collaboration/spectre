// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <ostream>

#include "NumericalAlgorithms/Interpolation/CubicSpline.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

/// \cond
namespace Options {
struct Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace RelativisticEuler::Solutions {

/// Radial coordinate of a TOV solution.
enum class TovCoordinates {
  /// Areal radial coordinate, in which the exterior TOV solution coincides with
  /// the Schwarzschild metric in standard Schwarzschild coordinates.
  Schwarzschild,
  /// Isotropic radial coordinate, in which the spatial metric is conformally
  /// flat.
  Isotropic
};

std::ostream& operator<<(std::ostream& os, TovCoordinates coords);

}  // namespace RelativisticEuler::Solutions

template <>
struct Options::create_from_yaml<RelativisticEuler::Solutions::TovCoordinates> {
  template <typename Metavariables>
  static RelativisticEuler::Solutions::TovCoordinates create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
RelativisticEuler::Solutions::TovCoordinates
Options::create_from_yaml<RelativisticEuler::Solutions::TovCoordinates>::create<
    void>(const Options::Option& options);

namespace RelativisticEuler::Solutions {

/*!
 * \brief TOV solver based on Lindblom's method
 *
 * Uses Lindblom's method of integrating the TOV equations from
 * \cite Lindblom1998dp .
 *
 * Instead of integrating the interior mass \f$m(r)\f$ and pressure \f$p(r)\f$,
 * Lindblom introduces the variables \f$u=r^2\f$ and \f$v=m/r\f$. Then, the TOV
 * equations are integrated with the log of the specific enthalpy as the
 * independent variable, \f$\ln(h)\f$, from the center of the star where
 * \f$h(r=0) = h_c\f$ to its surface \f$h(r=R) = 1\f$. The ODEs being solved are
 * Eq. (A2) and (A3) in \cite Lindblom1998dp :
 *
 * \f{align}
 * \frac{\mathrm{d}u}{\mathrm{d}\ln{h}} &= \frac{-2u (1 - 2v)}{4\pi u p + v} \\
 * \frac{\mathrm{d}v}{\mathrm{d}\ln{h}} &=
 *   -(1 - 2v) \frac{4\pi u \rho - v}{4\pi u p + v}
 * \f}
 *
 * Note that Lindblom's paper labels the independent variable as \f$h\f$.
 * However the \f$h\f$ in Lindblom's paper is **not** the specific enthalpy but
 * its logarithm, \f$\ln(h)\f$.
 *
 * The ODEs are solved numerically when this class is constructed, and the
 * quantities \f$m(r)/r\f$ and \f$\ln(h)\f$ are interpolated and exposed as
 * member functions. With these quantities the metric can be constructed as:
 *
 * \f{equation}
 * \mathrm{d}s^2 = -\alpha^2 \mathrm{d}t^2 + (1 - 2m/r)^{-1} \mathrm{d}r^2
 *   + r^2 \mathrm{d}\Omega^2
 * \f}
 *
 * where the lapse is
 *
 * \f{equation}
 * \alpha(r < R) = \mathcal{E} / h = \alpha(r=R) / h
 * \text{,}
 * \f}
 *
 * with the conserved `injection_energy()` \f$\mathcal{E}\f$, such that the
 * lapse matches the exterior Schwarzschild solution:
 *
 * \f{equation}
 * \alpha(r \geq R) = \sqrt{1 - \frac{2M}{r}}
 * \f}
 *
 * \par Isotropic radial coordinate
 * This class also supports transforming to an isotropic radial coordinate. When
 * you pass `RelativisticEuler::Solutions::TovCoordinates::Isotropic` to the
 * constructor, an additional ODE is integrated alongside the TOV equations to
 * determine the conformal factor
 *
 * \f{equation}
 * \psi^2 = \frac{r}{\bar{r}}
 * \f}
 *
 * where \f$r\f$ is the areal (Schwarzschild) radius and \f$\bar{r}\f$ is the
 * isotropic radius. The additional ODE is:
 *
 * \f{equation}
 * \frac{\mathrm{d}\ln(\psi)}{\mathrm{d}\ln{h}} =
 *   \frac{\sqrt{1 - 2v}}{1 + \sqrt{1 - 2v}} \frac{v}{4\pi u p + v}
 * \f}
 *
 * In isotropic coordinates, the spatial metric is conformally flat:
 *
 * \f{equation}
 * \mathrm{d}s^2 = -\alpha^2 \mathrm{d}t^2 + \psi^4 (\mathrm{d}\bar{r}^2 +
 *   \bar{r}^2 \mathrm{d}\Omega^2)
 * \f}
 *
 * When isotropic coordinates are selected, radii returned by member functions
 * or taken as arguments are isotropic. An exception is `mass_over_radius()`,
 * which always returns the quantity \f$m / r\f$ because that is the quantity
 * stored internally and hence most numerically precise. See
 * `mass_over_radius()` for details.
 */
class TovSolution {
 public:
  TovSolution(
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
      double central_mass_density,
      const TovCoordinates coordinate_system = TovCoordinates::Schwarzschild,
      double log_enthalpy_at_outer_radius = 0.0,
      double absolute_tolerance = 1.e-18, double relative_tolerance = 1.0e-14);

  TovSolution() = default;
  TovSolution(const TovSolution& /*rhs*/) = default;
  TovSolution& operator=(const TovSolution& /*rhs*/) = default;
  TovSolution(TovSolution&& /*rhs*/) = default;
  TovSolution& operator=(TovSolution&& /*rhs*/) = default;
  ~TovSolution() = default;

  /// The type of radial coordinate.
  ///
  /// \see RelativisticEuler::Solutions::TovCoordinates
  TovCoordinates coordinate_system() const { return coordinate_system_; }

  /// \brief The outer radius of the solution.
  ///
  /// This is the outer radius in the specified `coordinate_system()`, i.e.,
  /// areal or isotropic.
  ///
  /// \note This is the radius at which `log_specific_enthalpy` is equal
  /// to the value of `log_enthalpy_at_outer_radius` that was given when
  /// constructing this TovSolution
  double outer_radius() const { return outer_radius_; }

  /// The total mass \f$m(R)\f$, where \f$R\f$ is the outer radius.
  double total_mass() const { return total_mass_; }

  /*!
   * \brief The injection energy \f$\mathcal{E}=\alpha(r=R)=\sqrt{1-2M/R}\f$.
   *
   * The injection energy of the TOV solution is
   *
   * \f{equation}
   * \mathcal{E} = -h k^a u_a = h \alpha
   * \text{,}
   * \f}
   *
   * where \f$\boldsymbol{k} = \partial_t\f$ is a Killing vector of the static
   * solution, \f$h\f$ is the specific enthalpy, \f$u_a\f$ is the fluid
   * four-velocity, and \f$\alpha\f$ is the lapse (see, e.g., Eqs. (2.19) and
   * (4.2) in \cite Moldenhauer2014yaa). Since the TOV solution is static, the
   * injection energy is conserved not only along stream lines but throughout
   * the star,
   *
   * \f{equation}
   * \nabla_a \mathcal{E} = 0
   * \text{.}
   * \f}
   *
   * Therefore,
   *
   * \f{equation}
   * \mathcal{E} = \alpha(r=R) = \sqrt{1 - 2M/R}
   * \f}
   *
   * by evaluating the injection energy at the outer (areal) radius \f$R\f$,
   * where \f$h=1\f$ and where we match the lapse to the outer Schwarzschild
   * solution. The conservation also implies
   *
   * \f{equation}
   * \alpha = \mathcal{E} / h
   * \f}
   *
   * throughout the star.
   */
  double injection_energy() const { return injection_energy_; }

  /// \brief The mass inside the given radius over the areal radius,
  /// \f$\frac{m(r)}{r}\f$
  ///
  /// The argument to this function is the radius in the `coordinate_system()`,
  /// i.e., areal (Schwarzschild) or isotropic radius. The denominator \f$r\f$
  /// in the return value is always the areal (Schwarzschild) radius. You can
  /// use the conformal factor \f$\psi=\sqrt{r / \bar{r}}\f$ returned by the
  /// `conformal_factor()` function to obtain the mass over the isotropic
  /// radius, or the mass alone. The reason for this choice is that we represent
  /// the solution internally as the mass over the areal radius, so this is the
  /// most numerically precise quantity from which other quantities can be
  /// derived.
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`.
  template <typename DataType>
  DataType mass_over_radius(const DataType& r) const;

  /// \brief The log of the specific enthalpy at the given radius
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`
  template <typename DataType>
  DataType log_specific_enthalpy(const DataType& r) const;

  /// \brief The conformal factor \f$\psi=\sqrt{r / \bar{r}}\f$.
  ///
  /// The conformal factor is computed only when the `coordinate_system()` is
  /// `RelativisticEuler::Solution::TovCoordinates::Isotropic`. Otherwise, it is
  /// an error to call this function.
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`
  template <typename DataType>
  DataType conformal_factor(const DataType& r) const;

  const intrp::CubicSpline& mass_over_radius_interpolant() const {
    return mass_over_radius_interpolant_;
  }
  const intrp::CubicSpline& log_specific_enthalpy_interpolant() const {
    return log_enthalpy_interpolant_;
  }
  const intrp::CubicSpline& conformal_factor_interpolant() const {
    return conformal_factor_interpolant_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <TovCoordinates CoordSystem>
  void integrate(
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
      const double central_mass_density,
      const double log_enthalpy_at_outer_radius,
      const double absolute_tolerance, const double relative_tolerance);

  TovCoordinates coordinate_system_{};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double total_mass_{std::numeric_limits<double>::signaling_NaN()};
  double injection_energy_{std::numeric_limits<double>::signaling_NaN()};
  intrp::CubicSpline mass_over_radius_interpolant_;
  intrp::CubicSpline log_enthalpy_interpolant_;
  intrp::CubicSpline conformal_factor_interpolant_;
};

}  // namespace RelativisticEuler::Solutions
