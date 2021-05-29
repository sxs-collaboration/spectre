// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/Systems/Burgers/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::BoundaryConditions {
class Outflow final : public BoundaryCondition {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Outflow boundary condition that only verifies the characteristic speeds "
      "are all directed out of the domain."};

  Outflow() = default;
  Outflow(Outflow&&) noexcept = default;
  Outflow& operator=(Outflow&&) noexcept = default;
  Outflow(const Outflow&) = default;
  Outflow& operator=(const Outflow&) = default;
  ~Outflow() override = default;

  explicit Outflow(CkMigrateMessage* msg) noexcept;

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Outflow);

  auto get_clone() const noexcept -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  static constexpr evolution::BoundaryConditions::Type bc_type =
      evolution::BoundaryConditions::Type::Outflow;

  void pup(PUP::er& p) override;

  using dg_interior_evolved_variables_tags = tmpl::list<Tags::U>;
  using dg_interior_temporary_tags = tmpl::list<>;
  using dg_gridless_tags = tmpl::list<>;

  static std::optional<std::string> dg_outflow(
      const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
          face_mesh_velocity,
      const tnsr::i<DataVector, 1, Frame::Inertial>&
          outward_directed_normal_covector,
      const Scalar<DataVector>& u) noexcept;
};
}  // namespace Burgers::BoundaryConditions
