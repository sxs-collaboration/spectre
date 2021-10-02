// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
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
  TovSolution(const TovSolution& /*rhs*/) = delete;
  TovSolution& operator=(const TovSolution& /*rhs*/) = delete;
  TovSolution(TovSolution&& /*rhs*/) = default;
  TovSolution& operator=(TovSolution&& /*rhs*/) = default;
  ~TovSolution() = default;

  /// \brief The outer radius of the solution.
  ///
  /// \note This is the radius at which `log_specific_enthalpy` is equal
  /// to the value of `log_enthalpy_at_outer_radius` that was given when
  /// constructing this TovSolution
  double outer_radius() const;

  /// \brief The mass inside the given radius over the radius
  /// \f$\frac{m(r)}{r}\f$
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`.
  double mass_over_radius(double r) const;

  /// \brief The mass inside the given radius \f$m(r)\f$
  ///
  /// \warning When computing \f$\frac{m(r)}{r}\f$, use the `mass_over_radius`
  /// function instead for greater accuracy.
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`
  double mass(double r) const;

  /// \brief The log of the specific enthalpy at the given radius
  ///
  /// \note `r` should be non-negative and not greater than `outer_radius()`
  double log_specific_enthalpy(double r) const;

  /// \brief The radial variables from which the hydrodynamic quantities and
  /// spacetime metric can be computed.
  ///
  /// For radii greater than the `outer_radius()`, this returns the appropriate
  /// vacuum spacetime.
  ///
  /// \note This solution of the TOV equations is a function of areal radius.
  template <typename DataType>
  RelativisticEuler::Solutions::TovStar<TovSolution>::RadialVariables<DataType>
  radial_variables(
      const EquationsOfState::EquationOfState<true, 1>& equation_of_state,
      const tnsr::I<DataType, 3>& x) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double total_mass_{std::numeric_limits<double>::signaling_NaN()};
  double log_lapse_at_outer_radius_{
      std::numeric_limits<double>::signaling_NaN()};
  intrp::BarycentricRational mass_over_radius_interpolant_;
  intrp::BarycentricRational log_enthalpy_interpolant_;
};

}  // namespace Solutions
}  // namespace gr
