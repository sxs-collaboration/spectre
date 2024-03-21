// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/IrrotationalBns/BoundaryConditions/Neumann.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/TMPL.hpp"

namespace IrrotationalBns::BoundaryConditions {

namespace {
template <bool Linearized>
void apply_neumann_boundary_condition(
    const gsl::not_null<Scalar<DataVector>*> n_dot_auxilliary_velocity) {
  const Neumann boundary_condition{{}};
  Scalar<DataVector> field{DataVector{1.0, 2.0, 3.0, 4.0}};
  const auto box = db::create<db::AddSimpleTags<>>();
  elliptic::apply_boundary_condition<Linearized, void, tmpl::list<Neumann>>(
      boundary_condition, box, Direction<3>::lower_xi(), make_not_null(&field),
      n_dot_auxilliary_velocity);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IrrotationalBns.BoundaryConditions.Neumann",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, Neumann>("Neumann:");
  REQUIRE(dynamic_cast<const Neumann*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Neumann&>(*created);
  {
    {
      INFO("Semantics");
      test_serialization(boundary_condition);
      test_copy_semantics(boundary_condition);
      auto move_boundary_condition = boundary_condition;
      test_move_semantics(std::move(move_boundary_condition),
                          boundary_condition);
    }
    {
      Scalar<DataVector> n_dot_auxilliary_velocity{};
      Scalar<DataVector> n_dot_auxilliary_velocity_for_linearized{};
      {
        apply_neumann_boundary_condition<true>(
            make_not_null(&n_dot_auxilliary_velocity_for_linearized));
        CHECK(get(n_dot_auxilliary_velocity_for_linearized) ==
              DataVector{4, 0.0});
      }
      {
        apply_neumann_boundary_condition<false>(
            make_not_null(&n_dot_auxilliary_velocity));
        CHECK(get(n_dot_auxilliary_velocity) == DataVector{4, 0.0});
      }
    }
  }
}
}  // namespace IrrotationalBns::BoundaryConditions
