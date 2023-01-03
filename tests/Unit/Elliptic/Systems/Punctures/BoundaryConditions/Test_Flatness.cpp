// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Punctures/BoundaryConditions/Flatness.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures::BoundaryConditions {

namespace {
template <bool Linearized>
void apply_boundary_condition(
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient,
    Scalar<DataVector> field, tnsr::I<DataVector, 3> x) {
  const Flatness boundary_condition{};
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{
          {Direction<3>::upper_xi(), std::move(x)}});
  // grad(u) will be set by the boundary condition
  elliptic::apply_boundary_condition<Linearized, void, tmpl::list<Flatness>>(
      boundary_condition, box, Direction<3>::upper_xi(), make_not_null(&field),
      n_dot_field_gradient);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Punctures.BoundaryConditions.Flatness",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created =
      TestHelpers::test_factory_creation<
          elliptic::BoundaryConditions::BoundaryCondition<3>, Flatness>(
          "Flatness")
          ->get_clone();
  REQUIRE(dynamic_cast<const Flatness*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Flatness&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Neumann});
  }
  {
    INFO("Random-value tests");
    pypp::SetupLocalPythonEnvironment local_python_env(
        "Elliptic/Systems/Punctures/BoundaryConditions/");
    pypp::check_with_random_values<2>(&apply_boundary_condition<false>,
                                      "Flatness", {"normal_dot_field_gradient"},
                                      {{{-1., 1.}, {-1., 1.}}}, DataVector{3});
    pypp::check_with_random_values<2>(&apply_boundary_condition<true>,
                                      "Flatness", {"normal_dot_field_gradient"},
                                      {{{-1., 1.}, {-1., 1.}}}, DataVector{3});
  }
}

}  // namespace Punctures::BoundaryConditions
