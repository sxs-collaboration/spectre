// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP

namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags

class DataVector;
template <typename>
class Variables;
/// \endcond

namespace NewtonianEuler {
namespace NumericalFluxes {

/*!
 * \brief Compute the HLLC numerical flux
 *
 * Class implementing the HLLC flux for the Newtonian Euler equations,
 * originally introduced by E. F. Toro, M. Spruce and W. Speares \cite Toro1994.
 * Let \f$F^k = [F^k(\rho), F^k(S^i), F^k(e)]^T\f$ be the fluxes of
 * the conservative quantities \f$U = [\rho, S^i, e]^T\f$
 * (see NewtonianEuler::ComputeFluxes). Denoting \f$v_n = n_k v^k\f$ and
 * \f$F = n_kF^k\f$, where \f$v^k\f$ is the velocity and
 * \f$n_k\f$ is the interface unit normal, the HLLC flux is
 *
 * \f{align*}
 * G_\text{HLLC} =
 * \begin{cases}
 * F_\text{int}, & S_\text{min} > 0, \\
 * G_{*\text{int}}, & S_\text{min} \leq 0\quad\text{and}\quad S_* > 0, \\
 * G_{*\text{ext}}, & S_* \leq 0\quad\text{and}\quad S_\text{max} > 0, \\
 * F_\text{ext}, & S_\text{max} \leq 0,
 * \end{cases}
 * \f}
 *
 * where
 *
 * \f{align*}
 * G_{*\text{int}} &= \frac{S_*\left(F_\text{int}
 * - S_\text{min}U_\text{int}\right)
 * - S_\text{min} p_{*\text{int}}D_*}{S_* - S_\text{min}},\\
 * G_{*\text{ext}} &= \frac{S_*\left(F_\text{ext}
 * - S_\text{max}U_\text{ext}\right)
 * - S_\text{max} p_{*\text{ext}}D_*}{S_* - S_\text{max}},
 * \f}
 *
 * with
 *
 * \f{align*}
 * p_{*\text{int}} &\equiv p_\text{int} + \rho_\text{int}
 * \left[(v_n)_\text{int} - S_\text{min}\right]
 * \left[(v_n)_\text{int} - S_*\right], \\
 * p_{*\text{ext}} &\equiv p_\text{ext} + \rho_\text{ext}
 * \left[(v_n)_\text{ext} - S_\text{max}\right]
 * \left[(v_n)_\text{ext} - S_*\right], \\
 * S_* &\equiv \frac{p_\text{int} + F(\rho)_\text{int}\left[(v_n)_\text{int}
 * - S_\text{min}\right] - p_\text{ext}
 * - F(\rho)_\text{ext}\left[(v_n)_\text{ext} -
 * S_\text{max}\right]}{F(\rho)_\text{int} - \rho_\text{int}S_\text{min}
 * - F(\rho)_\text{ext} + \rho_\text{ext}S_\text{max}},\\
 * D_* &\equiv \left[\begin{array}{c}
 * 0\\ n^i\\ S_*
 * \end{array}\right],
 * \f}
 *
 * and \f$S_\text{min}\f$ and \f$S_\text{max}\f$ are estimates of the minimum
 * and maximum signal speeds bounding the ingoing and outgoing
 * wavespeeds that arise when solving the Riemann problem. One requires
 * \f$S_\text{min} \leq S_* \leq S_\text{max}\f$. As estimates, we use
 *
 * \f{align*}
 * S_\text{min} &=
 * \text{min}\left(\{\lambda_\text{int}\},\{\lambda_\text{ext}\}, 0\right)\\
 * S_\text{max} &=
 * \text{max}\left(\{\lambda_\text{int}\},\{\lambda_\text{ext}\}, 0\right),
 * \f}
 *
 * where \f$\{\lambda\}\f$ is the set of all the characteristic speeds along a
 * given normal. This way, the definition of \f$G_\text{HLLC}\f$ simplifies to
 *
 * \f{align*}
 * G_\text{HLLC} =
 * \begin{cases}
 * \dfrac{S_*\left(F_\text{int} - S_\text{min}U_\text{int}\right)
 * - S_\text{min} p_{*\text{int}}D_*}{S_* - S_\text{min}}, & S_* > 0, \\
 * \dfrac{S_*\left(F_\text{ext} - S_\text{max}U_\text{ext}\right)
 * - S_\text{max} p_{*\text{ext}}D_*}{S_* - S_\text{max}}, & S_* \leq 0. \\
 * \end{cases}
 * \f}
 *
 * \warning The HLLC flux implemented here does not incorporate any cure for
 * the Carbuncle phenomenon or other shock instabilities reported in the
 * literature. Prefer using another numerical flux in more than 1-d.
 */
template <size_t Dim, typename Frame>
struct Hllc {
  using char_speeds_tag = Tags::CharacteristicSpeedsCompute<Dim>;

  /// Estimate for one of the signal speeds
  struct LargestIngoingSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  /// Estimate for the other signal speed
  struct LargestOutgoingSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  /// The normal component of the velocity
  struct NormalVelocity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Compute the HLLC flux for the Newtonian Euler system."};

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  using package_tags = tmpl::list<
      ::Tags::NormalDotFlux<Tags::MassDensityCons>,
      ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim, Frame>>,
      ::Tags::NormalDotFlux<Tags::EnergyDensity>, Tags::MassDensityCons,
      Tags::MomentumDensity<Dim, Frame>, Tags::EnergyDensity,
      Tags::Pressure<DataVector>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>,
      NormalVelocity, LargestIngoingSpeed, LargestOutgoingSpeed>;

  using argument_tags = tmpl::list<
      ::Tags::NormalDotFlux<Tags::MassDensityCons>,
      ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim, Frame>>,
      ::Tags::NormalDotFlux<Tags::EnergyDensity>, Tags::MassDensityCons,
      Tags::MomentumDensity<Dim, Frame>, Tags::EnergyDensity,
      Tags::Velocity<DataVector, Dim, Frame>, Tags::Pressure<DataVector>,
      char_speeds_tag,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_mass_density,
      const tnsr::I<DataVector, Dim, Frame>& normal_dot_flux_momentum_density,
      const Scalar<DataVector>& normal_dot_flux_energy_density,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim, Frame>& velocity,
      const Scalar<DataVector>& pressure,
      const db::const_item_type<char_speeds_tag>& characteristic_speeds,
      const tnsr::i<DataVector, Dim, Frame>& interface_unit_normal) const
      noexcept;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame>*>
          normal_dot_numerical_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          normal_dot_numerical_flux_energy_density,
      const Scalar<DataVector>& normal_dot_flux_mass_density_int,
      const tnsr::I<DataVector, Dim, Frame>&
          normal_dot_flux_momentum_density_int,
      const Scalar<DataVector>& normal_dot_flux_energy_density_int,
      const Scalar<DataVector>& mass_density_int,
      const tnsr::I<DataVector, Dim, Frame>& momentum_density_int,
      const Scalar<DataVector>& energy_density_int,
      const Scalar<DataVector>& pressure_int,
      const tnsr::i<DataVector, Dim, Frame>& interface_unit_normal,
      const Scalar<DataVector>& normal_velocity_int,
      const Scalar<DataVector>& largest_ingoing_speed_int,
      const Scalar<DataVector>& largest_outgoing_speed_int,
      const Scalar<DataVector>& minus_normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame>&
          minus_normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& minus_normal_dot_flux_energy_density_ext,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const Scalar<DataVector>& pressure_ext,
      const tnsr::i<DataVector, Dim, Frame>& minus_interface_unit_normal,
      const Scalar<DataVector>& minus_normal_velocity_ext,
      // names are inverted w.r.t interior data. See package_data()
      const Scalar<DataVector>& minus_largest_outgoing_speed_ext,
      const Scalar<DataVector>& minus_largest_ingoing_speed_ext) const noexcept;
};

}  // namespace NumericalFluxes
}  // namespace NewtonianEuler
