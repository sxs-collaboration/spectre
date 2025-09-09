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
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Sommerfeld.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.ScalarSelfForce.BoundaryConditions.Sommerfeld",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Sommerfeld>(
      "Sommerfeld:\n"
      "  BlackHoleMass: 1.0\n"
      "  BlackHoleSpin: 0.5\n"
      "  OrbitalRadius: 10.0\n"
      "  MModeNumber: 2\n"
      "  HyperboloidalSlicing: False\n");
  REQUIRE(dynamic_cast<const Sommerfeld*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Sommerfeld&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    CHECK(boundary_condition.black_hole_mass() == 1.0);
    CHECK(boundary_condition.black_hole_spin() == 0.5);
    CHECK(boundary_condition.orbital_radius() == 10.0);
    CHECK(boundary_condition.m_mode_number() == 2);
    CHECK(boundary_condition.hyperboloidal_slicing() == false);
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Neumann});
  }
  {
    INFO("Apply boundary condition");
    const DataVector used_for_size(5);
    auto field = make_with_value<Scalar<ComplexDataVector>>(used_for_size, 1.);
    auto n_dot_field_gradient = make_with_value<Scalar<ComplexDataVector>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    const tnsr::i<ComplexDataVector, 2> deriv_field{
        used_for_size.size(), std::numeric_limits<double>::signaling_NaN()};
    const auto box = db::create<db::AddSimpleTags<>>();
    elliptic::apply_boundary_condition<false, void,
                                       standard_boundary_conditions>(
        boundary_condition, box, Direction<1>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    const ComplexDataVector expected_value(
        used_for_size.size(), std::complex<double>(0., 0.06226111848298833));
    CHECK_ITERABLE_APPROX(get(n_dot_field_gradient), expected_value);
  }
}

}  // namespace ScalarSelfForce::BoundaryConditions
