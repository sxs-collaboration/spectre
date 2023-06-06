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
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/String.hpp"
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
class DemandOutgoingCharSpeeds final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A boundary condition that only verifies the characteristic speeds are "
      "all directed out of the domain."};

  DemandOutgoingCharSpeeds() = default;
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  ~DemandOutgoingCharSpeeds() override = default;

  explicit DemandOutgoingCharSpeeds(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, DemandOutgoingCharSpeeds);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::DemandOutgoingCharSpeeds;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_interior_primitive_variables_tags = tmpl::list<
      NewtonianEuler::Tags::MassDensity<DataVector>,
      NewtonianEuler::Tags::Velocity<DataVector, Dim, Frame::Inertial>,
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_gridless_tags = tmpl::list<hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          outward_directed_normal_covector,

      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);
};
}  // namespace NewtonianEuler::BoundaryConditions
