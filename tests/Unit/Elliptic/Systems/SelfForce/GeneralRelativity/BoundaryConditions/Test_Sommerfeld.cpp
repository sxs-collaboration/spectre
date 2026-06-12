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
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/BoundaryConditions/Sommerfeld.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.GrSelfForce.BoundaryConditions.Sommerfeld",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Sommerfeld>(
      "Sommerfeld:\n"
      "  BlackHoleMass: 1.1\n"
      "  BlackHoleSpin: 0.5\n"
      "  OrbitalRadius: 10.0\n"
      "  MModeNumber: 2\n"
      "  HyperboloidalSlicing: false\n"
      "  Order: 1\n");
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
    CHECK(boundary_condition.black_hole_mass() == 1.1);
    CHECK(boundary_condition.black_hole_spin() == 0.5);
    CHECK(boundary_condition.orbital_radius() == 10.0);
    CHECK(boundary_condition.m_mode_number() == 2);
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              elliptic::BoundaryConditionType::Neumann});
  }
  {
    INFO("Apply boundary condition");
    const DataVector used_for_size(5);
    auto field =
        make_with_value<tnsr::aa<ComplexDataVector, 3>>(used_for_size, 1.);
    auto n_dot_field_gradient = make_with_value<tnsr::aa<ComplexDataVector, 3>>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    using GradTensorType = TensorMetafunctions::prepend_spatial_index<
        tnsr::aa<ComplexDataVector, 3>, 2, UpLo::Lo, Frame::Inertial>;
    auto deriv_field = make_with_value<GradTensorType>(
        used_for_size, std::numeric_limits<double>::signaling_NaN());
    DirectionMap<2, tnsr::I<ComplexDataVector, 2>> alpha_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::I<ComplexDataVector, 2>>(used_for_size, 0.)}};
    DirectionMap<2, tnsr::aaBB<ComplexDataVector, 3>> beta_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::aaBB<ComplexDataVector, 3>>(used_for_size, 0.)}};
    DirectionMap<2, tnsr::aaBB<ComplexDataVector, 3>> gammarstar_map{
        {Direction<2>::lower_xi(),
         make_with_value<tnsr::aaBB<ComplexDataVector, 3>>(used_for_size, 0.)}};

    const auto box =
        db::create<db::AddSimpleTags<domain::Tags::Faces<2, Tags::Alpha>,
                                     domain::Tags::Faces<2, Tags::Beta>,
                                     domain::Tags::Faces<2, Tags::GammaRstar>>>(
            alpha_map, beta_map, gammarstar_map);
    elliptic::apply_boundary_condition<false, void,
                                       standard_boundary_conditions>(
        boundary_condition, box, Direction<2>::lower_xi(),
        make_not_null(&field), make_not_null(&n_dot_field_gradient),
        deriv_field);
    const int m_mode = 2;
    const double M = 1.1;
    const double r0 = 10.0;
    const double a = 0.5 * M;
    const double omega = m_mode / (a + sqrt(cube(r0) / M));
    tnsr::aa<ComplexDataVector, 3> expected_value{used_for_size.size()};
    for (auto& component : expected_value) {
      component = std::complex<double>(0., omega);
    }
    CHECK_ITERABLE_APPROX(n_dot_field_gradient, expected_value);
  }
}

}  // namespace GrSelfForce::BoundaryConditions
