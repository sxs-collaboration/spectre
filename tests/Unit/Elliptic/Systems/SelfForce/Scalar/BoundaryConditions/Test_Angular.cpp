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
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Angular.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Factory.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.ScalarSelfForce.BoundaryConditions.Angular",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Angular>(
      "Angular:\n"
      "  MModeNumber: 2\n");
  REQUIRE(dynamic_cast<const Angular*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Angular&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    CHECK(boundary_condition.m_mode_number() == 2);
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Dirichlet});
  }
  {
    INFO("Apply boundary condition");
    const DataVector used_for_size(5);
    auto field = make_with_value<Scalar<ComplexDataVector>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    auto n_dot_field_gradient = make_with_value<Scalar<ComplexDataVector>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    const tnsr::i<ComplexDataVector, 2> deriv_field{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    const auto box = db::create<db::AddSimpleTags<>>();
    using local_angular_list = tmpl::list<Angular>;
    elliptic::apply_boundary_condition<
        false, void, local_angular_list>(
        boundary_condition, box, Direction<2>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    const ComplexDataVector expected_value(used_for_size.size(), 0.);
    CHECK_ITERABLE_APPROX(get(field), expected_value);
  }
}

}  // namespace ScalarSelfForce::BoundaryConditions
