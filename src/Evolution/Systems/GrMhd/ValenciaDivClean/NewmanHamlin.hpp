// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <cstddef>
#include <string>

#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

namespace grmhd {
namespace ValenciaDivClean {
namespace PrimitiveRecoverySchemes {

/// \cond
struct PrimitiveRecoveryData;
/// \endcond

/*!
 * \brief Compute the primitive variables from the conservative variables using
 * the scheme of [Newman and Hamlin, SIAM J. Sci. Comput., 36(4)
 * B661-B683 (2014)](https://epubs.siam.org/doi/10.1137/140956749).
 *
 * In the Newman and Hamlin paper, `total_energy_density` is \f$e\f$,
 * `momentum_density_squared` is\f${\cal M}^2\f$,
 * `momentum_density_dot_magnetic_field` is \f${\cal T}\f$,
 * `magnetic_field_squared` is \f${\cal B}^2\f$, and
 * `rest_mass_density_times_lorentz_factor` is \f${\tilde \rho}\f$.
 * Furthermore, the returned `PrimitiveRecoveryData.rho_h_w_squared` is \f${\cal
 * L}\f$.
 *
 * In terms of the conservative variables (in our notation):
 * \f{align}
 * e = & \frac{{\tilde D} + {\tilde \tau}}{\sqrt{\gamma}} \\
 * {\cal M}^2 = & \frac{\gamma^{mn} {\tilde S}_m {\tilde S}_n}{\gamma} \\
 * {\cal T} = & \frac{{\tilde B}^m {\tilde S}_m}{\gamma} \\
 * {\cal B}^2 = & \frac{\gamma_{mn} {\tilde B}^m {\tilde B}^n}{\gamma} \\
 * {\tilde \rho} = & \frac{\tilde D}{\sqrt{\gamma}}
 * \f}
 *
 * where the conserved variables \f${\tilde D}\f$, \f${\tilde S}_i\f$,
 * \f${\tilde \tau}\f$, and \f${\tilde B}^i\f$ are a generalized mass-energy
 * density, momentum density, specific internal energy density, and magnetic
 * field, and \f$\gamma\f$ and \f$\gamma^{mn}\f$ are the determinant and inverse
 * of the spatial metric \f$\gamma_{mn}\f$.
 */
class NewmanHamlin {
 public:
  template <size_t ThermodynamicDim>
  static boost::optional<PrimitiveRecoveryData> apply(
      double initial_guess_for_pressure, double total_energy_density,
      double momentum_density_squared,
      double momentum_density_dot_magnetic_field, double magnetic_field_squared,
      double rest_mass_density_times_lorentz_factor,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) noexcept;

  static const std::string name() noexcept { return "Newman Hamlin"; }

 private:
  static constexpr size_t max_iterations_ = 50;
  static constexpr double relative_tolerance_ = 1.e-10;
};
}  // namespace PrimitiveRecoverySchemes
}  // namespace ValenciaDivClean
}  // namespace grmhd
