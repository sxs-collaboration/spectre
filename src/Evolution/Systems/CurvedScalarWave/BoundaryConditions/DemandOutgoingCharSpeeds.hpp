// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags
/// \endcond

namespace CurvedScalarWave::BoundaryConditions {
/// A `BoundaryCondition` that only verifies that all characteristic speeds are
/// directed out of the domain; no boundary data is altered by this boundary
/// condition.
template <size_t Dim>
class DemandOutgoingCharSpeeds final : public BoundaryCondition<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Boundary conditions which check that all characteristic "
      "fields are outflowing."};
  DemandOutgoingCharSpeeds() = default;
  /// \cond
  DemandOutgoingCharSpeeds(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds& operator=(DemandOutgoingCharSpeeds&&) = default;
  DemandOutgoingCharSpeeds(const DemandOutgoingCharSpeeds&) = default;
  DemandOutgoingCharSpeeds& operator=(const DemandOutgoingCharSpeeds&) =
      default;
  /// \endcond
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
  using dg_interior_temporary_tags =
      tmpl::list<Tags::ConstraintGamma1, gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, Dim>>;
  using dg_interior_dt_vars_tags = tmpl::list<>;
  using dg_interior_deriv_vars_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  std::optional<std::string> dg_demand_outgoing_char_speeds(
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, Dim>& normal_covector,
      const tnsr::I<DataVector, Dim>& /*normal_vector*/,
      const Scalar<DataVector>& gamma1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift) const;
};
}  // namespace CurvedScalarWave::BoundaryConditions
