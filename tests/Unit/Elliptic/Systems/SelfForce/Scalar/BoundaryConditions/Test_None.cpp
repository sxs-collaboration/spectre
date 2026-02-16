// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/None.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.ScalarSelfForce.BoundaryConditions.None",
                  "[Unit][Elliptic]") {
  // 1. Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, None>("None");
  REQUIRE(dynamic_cast<const None*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const None&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Apply boundary condition");
    const DataVector used_for_size(5);
    auto field = make_with_value<Scalar<ComplexDataVector>>(used_for_size, 1.2);
    auto n_dot_field_gradient = make_with_value<Scalar<ComplexDataVector>>(
        used_for_size, 3.4);
    const tnsr::i<ComplexDataVector, 2> deriv_field{
        used_for_size.size(), 5.6};
    const auto box = db::create<db::AddSimpleTags<>>();
    elliptic::apply_boundary_condition<
        false, void, standard_boundary_conditions>(
        boundary_condition, box, Direction<2>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    CHECK(get(field) == ComplexDataVector(5, 1.2));
    CHECK(get(n_dot_field_gradient) == ComplexDataVector(5, 3.4));
  }
}

}  // namespace ScalarSelfForce::BoundaryConditions
