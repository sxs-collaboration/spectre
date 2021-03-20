// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryCorrections/BoundaryCorrection.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler::BoundaryCorrections {
/*!
 * \brief An HLL (Harten-Lax-van Leer) Riemann solver for NewtonianEuler system
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be the
 * outward directed unit normal to the interface. Denoting \f$F := n_i F^i\f$,
 * the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 *   \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior.
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ are defined as
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right)
 * \f}
 *
 * where \f$\lambda^{+}\f$ (\f$\lambda^{-}\f$) is the largest characteristic
 * speed in the outgoing (ingoing) direction. Note the minus signs in front of
 * \f$\lambda^{\pm}_\text{ext}\f$, which is because an outgoing speed w.r.t. the
 * neighboring element is an ingoing speed w.r.t. the local element, and vice
 * versa. Similarly, the \f$F_{\text{ext}}\f$ term in \f$G_\text{HLL}\f$ has a
 * positive sign because the outward directed normal of the neighboring element
 * has the opposite sign, i.e. \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * For the NewtonianEuler system, \f$\lambda^\pm\f$ are given as
 *
 * \f{align*}
 * \lambda^\pm = v^in_i \pm c_s
 * \f}
 *
 * where \f$v^i\f$ is the spatial velocity and \f$c_s\f$ the sound speed.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns \f$G -
 *   F_\text{int}\f$
 * - For either \f$\lambda_\text{min} = 0\f$ or \f$\lambda_\text{max} = 0\f$
 *   (i.e. all characteristics move in the same direction) the HLL boundary
 *   correction reduces to pure upwinding.
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 *   signal/characteristic speeds
 */
template <size_t Dim>
class Hll final : public BoundaryCorrection<Dim> {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the HLL boundary correction term for the "
      "Newtonian Euler/hydrodynamics system."};

  Hll() = default;
  Hll(const Hll&) = default;
  Hll& operator=(const Hll&) = default;
  Hll(Hll&&) = default;
  Hll& operator=(Hll&&) = default;
  ~Hll() override = default;

  /// \cond
  explicit Hll(CkMigrateMessage* msg) noexcept;
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hll);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const noexcept override;

  using dg_package_field_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity,
                 ::Tags::NormalDotFlux<Tags::MassDensityCons>,
                 ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim>>,
                 ::Tags::NormalDotFlux<Tags::EnergyDensity>,
                 LargestOutgoingCharSpeed, LargestIngoingCharSpeed>;
  using dg_package_data_temporary_tags = tmpl::list<>;
  using dg_package_data_primitive_tags =
      tmpl::list<NewtonianEuler::Tags::Velocity<DataVector, Dim>,
                 NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_volume_tags =
      tmpl::list<hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_momentum_density,
      gsl::not_null<Scalar<DataVector>*> packaged_energy_density,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          packaged_normal_dot_flux_momentum_density,
      gsl::not_null<Scalar<DataVector>*>
          packaged_normal_dot_flux_energy_density,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& energy_density,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_mass_density,
      const tnsr::IJ<DataVector, Dim, Frame::Inertial>& flux_momentum_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_energy_density,

      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,

      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) const noexcept;

  void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_momentum_density,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_energy_density,
      const Scalar<DataVector>& mass_density_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_int,
      const Scalar<DataVector>& energy_density_int,
      const Scalar<DataVector>& normal_dot_flux_mass_density_int,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_momentum_density_int,
      const Scalar<DataVector>& normal_dot_flux_energy_density_int,
      const Scalar<DataVector>& largest_outgoing_char_speed_int,
      const Scalar<DataVector>& largest_ingoing_char_speed_int,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      dg::Formulation dg_formulation) const noexcept;
};
}  // namespace NewtonianEuler::BoundaryCorrections
