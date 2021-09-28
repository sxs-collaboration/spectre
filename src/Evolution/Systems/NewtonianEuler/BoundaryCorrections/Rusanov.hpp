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
 * \brief A Rusanov/local Lax-Friedrichs Riemann solver
 *
 * Let \f$U\f$ be the state vector of evolved variables, \f$F^i\f$ the
 * corresponding fluxes, and \f$n_i\f$ be the outward directed unit normal to
 * the interface. Denoting \f$F := n_i F^i\f$, the %Rusanov boundary correction
 * is
 *
 * \f{align*}
 * G_\text{Rusanov} = \frac{F_\text{int} - F_\text{ext}}{2} -
 * \frac{\text{max}\left(\{|\lambda_\text{int}|\},
 * \{|\lambda_\text{ext}|\}\right)}{2} \left(U_\text{ext} - U_\text{int}\right),
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior, and
 * \f$\{|\lambda|\}\f$ is the set of characteristic/signal speeds. The minus
 * sign in front of the \f$F_{\text{ext}}\f$ is necessary because the outward
 * directed normal of the neighboring element has the opposite sign, i.e.
 * \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$. The characteristic/signal speeds
 * are given by:
 *
 * \f{align*}
 * \lambda_{\pm}&=v^i n_i \pm c_s, \\
 * \lambda_v&=v^i n_i
 * \f}
 *
 * where \f$v^i\f$ is the spatial velocity and \f$c_s\f$ the sound speed.
 *
 * \note In the strong form the `dg_boundary_terms` function returns
 * \f$G - F_\text{int}\f$
 */
template <size_t Dim>
class Rusanov final : public BoundaryCorrection<Dim> {
 private:
  struct AbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Computes the Rusanov or local Lax-Friedrichs boundary correction term "
      "for the Newtonian Euler/hydrodynamics system."};

  Rusanov() = default;
  Rusanov(const Rusanov&) = default;
  Rusanov& operator=(const Rusanov&) = default;
  Rusanov(Rusanov&&) = default;
  Rusanov& operator=(Rusanov&&) = default;
  ~Rusanov() override = default;

  /// \cond
  explicit Rusanov(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Rusanov);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection<Dim>> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>,
                 Tags::EnergyDensity,
                 ::Tags::NormalDotFlux<Tags::MassDensityCons>,
                 ::Tags::NormalDotFlux<Tags::MomentumDensity<Dim>>,
                 ::Tags::NormalDotFlux<Tags::EnergyDensity>, AbsCharSpeed>;
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
      gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,

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
          equation_of_state) const;

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
      const Scalar<DataVector>& abs_char_speed_int,
      const Scalar<DataVector>& mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density_ext,
      const Scalar<DataVector>& energy_density_ext,
      const Scalar<DataVector>& normal_dot_flux_mass_density_ext,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_momentum_density_ext,
      const Scalar<DataVector>& normal_dot_flux_energy_density_ext,
      const Scalar<DataVector>& abs_char_speed_ext,
      dg::Formulation dg_formulation) const;
};
}  // namespace NewtonianEuler::BoundaryCorrections
