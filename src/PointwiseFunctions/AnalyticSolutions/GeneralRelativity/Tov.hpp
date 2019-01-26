// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRational.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \cond
class DataVector;
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
  TovSolution(const std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>&
                  equation_of_state,
              double central_mass_density, double final_log_enthalpy,
              double absolute_tolerance = 1.0e-14,
              double relative_tolerance = 1.0e-14);

  TovSolution() = default;
  TovSolution(const TovSolution& /*rhs*/) = delete;
  TovSolution& operator=(const TovSolution& /*rhs*/) = delete;
  TovSolution(TovSolution&& /*rhs*/) noexcept = default;
  TovSolution& operator=(TovSolution&& /*rhs*/) noexcept = default;
  ~TovSolution() = default;

  double outer_radius() const noexcept;
  double mass(double r) const noexcept;
  double specific_enthalpy(double r) const noexcept;
  double log_specific_enthalpy(double r) const noexcept;

  Scalar<DataVector> mass(const Scalar<DataVector>& radius) const noexcept;
  Scalar<DataVector> specific_enthalpy(const Scalar<DataVector>& radius) const
      noexcept;
  Scalar<DataVector> log_specific_enthalpy(
      const Scalar<DataVector>& radius) const noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  intrp::BarycentricRational mass_interpolant_;
  intrp::BarycentricRational log_enthalpy_interpolant_;
};

}  // namespace Solutions
}  // namespace gr
