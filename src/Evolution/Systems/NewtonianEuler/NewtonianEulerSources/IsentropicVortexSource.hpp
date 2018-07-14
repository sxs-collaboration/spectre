// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace NewtonianEuler {
namespace Tags {
template <typename DataType>
struct MassDensity;
template <typename DataType, size_t Dim, typename VolumeFrame>
struct MomentumDensity;
template <typename DataType>
struct EnergyDensity;
template <typename DataType>
struct Pressure;
}  // namespace Tags
}  // namespace NewtonianEuler

class DataVector;
/// \endcond

namespace NewtonianEuler {
namespace Sources {

/*!
 * \brief Source generating a modified isentropic vortex.
 *
 * If Solutions::IsentropicVortex is modifed so that the flow velocity along
 * the \f$z-\f$axis is not a constant but a function of \f$z\f$, the new vortex
 * will be a solution to the 3-D Newtonian Euler equations with a source term,
 *
 * \f{align*}
 * \partial_t\rho + \partial_i F^i(\rho) &= S(\rho)\\
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i)\\
 * \partial_t e + \partial_i F^i(e) &= S(e),
 * \f}
 *
 * where \f$F^i(u)\f$ is the volume flux of the conserved quantity \f$u\f$
 * (see ComputeFluxes), and
 *
 * \f{align*}
 * S(\rho) &= \rho \dfrac{dv_z}{dz}\\
 * S(S_x) &= S_x \dfrac{dv_z}{dz}\\
 * S(S_y) &= S_y \dfrac{dv_z}{dz}\\
 * S(S_z) &= 2S_z \dfrac{dv_z}{dz}\\
 * S(e) &= \left(e + p + v_z S_z\right)\dfrac{dv_z}{dz},
 * \f}
 *
 * where \f$v_z = v_z(z)\f$ is the \f$z-\f$component of the flow velocity,
 * and \f$p\f$ is the pressure.
 */
struct IsentropicVortexSource {
  /// The \f$z-\f$component of the vortex flow velocity.
  struct VelocityAlongZ {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "VelocityAlongZ"; }
  };

  /// The derivative w.r.t. \f$z\f$ of the \f$z-\f$component of the velocity.
  struct DzVelocityAlongZ {
    using type = Scalar<DataVector>;
    static std::string name() noexcept { return "DzVelocityAlongZ"; }
  };

  using argument_tags =
      tmpl::list<Tags::MassDensity<DataVector>,
                 Tags::MomentumDensity<DataVector, 3, Frame::Inertial>,
                 Tags::EnergyDensity<DataVector>, Tags::Pressure<DataVector>,
                 VelocityAlongZ, DzVelocityAlongZ>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> mass_density_source,
      gsl::not_null<tnsr::I<DataVector, 3>*> momentum_density_source,
      gsl::not_null<Scalar<DataVector>*> energy_density_source,
      const Scalar<DataVector>& vortex_mass_density,
      const tnsr::I<DataVector, 3>& vortex_momentum_density,
      const Scalar<DataVector>& vortex_energy_density,
      const Scalar<DataVector>& vortex_pressure,
      const Scalar<DataVector>& vortex_velocity_z,
      const Scalar<DataVector>& dz_vortex_velocity_z) noexcept;
};

}  // namespace Sources
}  // namespace NewtonianEuler
