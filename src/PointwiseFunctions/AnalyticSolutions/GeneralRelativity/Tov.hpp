// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace gr {
namespace Solutions {

/*!
 * \brief TOV solver based on Lindblom's method
 *
 * Uses Lindblom's method of integrating the TOV equations from
 * \cite Lindblom1998dp
 *
 * Instead of integrating \f$m(r)\f$ and \f$p(r)\f$
 * (\f$r\f$=radius, \f$m\f$=mass, \f$p\f$=pressure)
 * Lindblom introduces the variables \f$u\f$ and \f$v\f$, with \f$u=r^{2}\f$ and
 * \f$v=m/r\f$.
 * The integration is then done with the log of the specific enthalpy
 * (\f$\mathrm{log}(h)\f$) as the independent variable.
 *
 * Lindblom's paper simply labels the independent variable as \f$h\f$.
 * The \f$h\f$ in Lindblom's paper is NOT the specific enthalpy.
 * Rather, Lindblom's \f$h\f$ is in fact \f$\mathrm{log}(h)\f$.
 */
class TovSolution {
 public:
  TovSolution(
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
      double central_mass_density, double log_enthalpy_at_outer_radius = 0.0,
      double absolute_tolerance = 1.0e-14, double relative_tolerance = 1.0e-14);

  TovSolution() = default;
  TovSolution(const TovSolution& /*rhs*/) = default;
  TovSolution& operator=(const TovSolution& /*rhs*/) = default;
  TovSolution(TovSolution&& /*rhs*/) = default;
  TovSolution& operator=(TovSolution&& /*rhs*/) = default;
  ~TovSolution() = default;

  /// \brief The outer radius of the solution.
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
   * by evaluating the injection energy at the outer radius \f$R\f$, where
   * \f$h=1\f$ and where we match the lapse to the outer Schwarzschild
   * solution. The conservation also implies
   *
   * \f{equation}
   * \alpha = \mathcal{E} / h
   * \f}
   *
   * throughout the star.
   */
  double injection_energy() const { return injection_energy_; }

  /// \brief The mass inside the given radius over the radius
  /// \f$\frac{m(r)}{r}\f$
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`.
  template <typename DataType>
  DataType mass_over_radius(const DataType& r) const;

  /// \brief The log of the specific enthalpy at the given radius
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`
  template <typename DataType>
  DataType log_specific_enthalpy(const DataType& r) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double total_mass_{std::numeric_limits<double>::signaling_NaN()};
  double injection_energy_{std::numeric_limits<double>::signaling_NaN()};
  intrp::BarycentricRational mass_over_radius_interpolant_;
  intrp::BarycentricRational log_enthalpy_interpolant_;
};

}  // namespace Solutions
}  // namespace gr
