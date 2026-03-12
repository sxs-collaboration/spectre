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
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/None.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.GrSelfForce.BoundaryConditions.None",
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
    auto field =
        make_with_value<tnsr::aa<ComplexDataVector, 3>>(used_for_size, 1.2);
    auto n_dot_field_gradient =
        make_with_value<tnsr::aa<ComplexDataVector, 3>>(used_for_size, 3.4);
    using GradTensorType = TensorMetafunctions::prepend_spatial_index<
        tnsr::aa<ComplexDataVector, 3>, 2, UpLo::Lo, Frame::Inertial>;
    auto deriv_field = make_with_value<GradTensorType>(used_for_size, 5.6);
    const auto box = db::create<db::AddSimpleTags<>>();
    using local_none_list = tmpl::list<None>;
    elliptic::apply_boundary_condition<false, void, local_none_list>(
        boundary_condition, box, Direction<2>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    for (size_t i = 0; i < field.size(); ++i) {
      CHECK(field[i] == ComplexDataVector(5, 1.2));
      CHECK(n_dot_field_gradient[i] == ComplexDataVector(5, 3.4));
    }
  }
}

}  // namespace GrSelfForce::BoundaryConditions
