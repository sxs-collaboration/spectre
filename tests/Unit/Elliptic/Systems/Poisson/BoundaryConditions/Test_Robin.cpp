// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Poisson/BoundaryConditions/Robin.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Poisson::BoundaryConditions {

namespace {
template <bool Linearized>
void apply_dirichlet_boundary_condition(
    const gsl::not_null<Scalar<DataVector>*> field,
    const double dirichlet_weight, const double constant,
    const Scalar<DataVector>& /*used_for_size*/) {
  const Robin<1> boundary_condition{dirichlet_weight, 0., constant};
  const auto box = db::create<db::AddSimpleTags<>>();
  auto n_dot_field_gradient = make_with_value<Scalar<DataVector>>(
      *field, std::numeric_limits<double>::signaling_NaN());
  elliptic::apply_boundary_condition<Linearized>(
      boundary_condition, box, Direction<1>::lower_xi(), field,
      make_not_null(&n_dot_field_gradient));
}
template <bool Linearized>
void apply_neumann_boundary_condition(
    const gsl::not_null<Scalar<DataVector>*> n_dot_field_gradient,
    Scalar<DataVector> field, const double dirichlet_weight,
    const double neumann_weight, const double constant) {
  const Robin<1> boundary_condition{dirichlet_weight, neumann_weight, constant};
  const auto box = db::create<db::AddSimpleTags<>>();
  elliptic::apply_boundary_condition<Linearized>(
      boundary_condition, box, Direction<1>::lower_xi(), make_not_null(&field),
      n_dot_field_gradient);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.Robin",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<
          1, tmpl::list<Registrars::Robin<1>>>>(
      "Robin:\n"
      "  DirichletWeight: 2.\n"
      "  NeumannWeight: 3.\n"
      "  Constant: 4.");
  REQUIRE(dynamic_cast<const Robin<1>*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Robin<1>&>(*created);
  {
    INFO("Properties");
    CHECK(boundary_condition.dirichlet_weight() == 2.);
    CHECK(boundary_condition.neumann_weight() == 3.);
    CHECK(boundary_condition.constant() == 4.);
  }
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Random-value tests")
    pypp::SetupLocalPythonEnvironment local_python_env(
        "Elliptic/Systems/Poisson/BoundaryConditions/");
    pypp::check_with_random_values<3>(
        &apply_dirichlet_boundary_condition<false>, "Robin",
        {"dirichlet_field"}, {{{0.5, 2.}, {-1., 1.}, {-1., 1.}}},
        DataVector{3});
    pypp::check_with_random_values<3>(&apply_dirichlet_boundary_condition<true>,
                                      "Robin", {"dirichlet_field_linearized"},
                                      {{{0.5, 2.}, {-1., 1.}, {-1., 1.}}},
                                      DataVector{3});
    pypp::check_with_random_values<4>(
        &apply_neumann_boundary_condition<false>, "Robin",
        {"neumann_normal_dot_field_gradient"},
        {{{-1., 1.}, {0.5, 2.}, {-1., 1.}, {-1., 1.}}}, DataVector{3});
    pypp::check_with_random_values<4>(
        &apply_neumann_boundary_condition<true>, "Robin",
        {"neumann_normal_dot_field_gradient_linearized"},
        {{{-1., 1.}, {0.5, 2.}, {-1., 1.}, {-1., 1.}}}, DataVector{3});
  }
}

}  // namespace Poisson::BoundaryConditions
