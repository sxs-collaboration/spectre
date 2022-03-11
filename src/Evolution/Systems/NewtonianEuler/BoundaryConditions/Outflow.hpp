// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler::BoundaryConditions {
/*!
 * \brief A boundary condition that only verifies that all characteristic speeds
 * are directed out of the domain; no boundary data is altered by this boundary
 * condition.
 */
template <size_t Dim>
class Outflow final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Outflow boundary condition that only verifies the characteristic speeds "
      "are all directed out of the domain."};

  Outflow() = default;
  Outflow(Outflow&&) = default;
  Outflow& operator=(Outflow&&) = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Outflow;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<
      NewtonianEuler::Tags::MassDensity<DataVector>,
      NewtonianEuler::Tags::Velocity<DataVector, Dim, Frame::Inertial>,
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_gridless_tags = tmpl::list<hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);

  using fd_interior_evolved_variables_tags = tmpl::list<>;
  using fd_interior_temporary_tags =
      tmpl::list<evolution::dg::subcell::Tags::Mesh<Dim>>;
  using fd_interior_primitive_variables_tags = tmpl::list<
      NewtonianEuler::Tags::MassDensity<DataVector>,
      NewtonianEuler::Tags::Velocity<DataVector, Dim, Frame::Inertial>,
      NewtonianEuler::Tags::Pressure<DataVector>,
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>;
  using fd_gridless_tags = tmpl::list<hydro::Tags::EquationOfStateBase,
                                      fd::Tags::Reconstructor<Dim>>;

  template <size_t ThermodynamicDim>
  static void fd_outflow(
      const gsl::not_null<Scalar<DataVector>*> mass_density,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
      const gsl::not_null<Scalar<DataVector>*> pressure,
      const Direction<Dim>& direction,

      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Mesh<Dim>& subcell_mesh,
      const Scalar<DataVector>& interior_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
      const Scalar<DataVector>& interior_pressure,
      const Scalar<DataVector>& interior_specific_internal_energy,

      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state,
      const fd::Reconstructor<Dim>& reconstructor);
};
}  // namespace NewtonianEuler::BoundaryConditions
