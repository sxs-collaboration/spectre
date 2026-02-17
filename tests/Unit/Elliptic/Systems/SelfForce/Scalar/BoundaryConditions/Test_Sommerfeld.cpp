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
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Factory.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/BoundaryConditions/Sommerfeld.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
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
  using BC = elliptic::BoundaryConditions::BoundaryCondition<2>;
  const std::string base_sommerfeld =
        "Sommerfeld:\n"
        "  BlackHoleMass: 1.0\n"
        "  BlackHoleSpin: 0.5\n"
        "  OrbitalRadius: 10.0\n"
        "  MModeNumber: 2\n";
  const Direction<2> direction = Direction<2>::lower_xi();
  const size_t num_pts = 5;
  {
    INFO("Testing Order 1");
    const auto created_1 = TestHelpers::test_factory_creation<BC, Sommerfeld>(
        base_sommerfeld + "  HyperboloidalSlicing: False\n  Order: 1");
    const auto& bc_1 = dynamic_cast<const Sommerfeld&>(*created_1);
    {
      INFO("Semantics");
      test_serialization(bc_1);
      test_copy_semantics(bc_1);
      auto move_bc_1 = bc_1;
      test_move_semantics(std::move(move_bc_1), bc_1);
    }
    auto field = make_with_value<Scalar<ComplexDataVector>>(num_pts, 1.0);
    auto n_dot_grad = make_with_value<Scalar<ComplexDataVector>>(num_pts, 0.0);
    const DirectionMap<2, tnsr::I<DataVector, 2>> coords_map{
        {direction, tnsr::I<DataVector, 2>{num_pts, 0.0}}};
    const DirectionMap<2, Scalar<ComplexDataVector>> beta_map{
        {direction, make_with_value<Scalar<ComplexDataVector>>(num_pts, 0.0)}};
    const DirectionMap<2, tnsr::i<ComplexDataVector, 2>> gamma_map{
        {direction,
         make_with_value<tnsr::i<ComplexDataVector, 2>>(num_pts, 0.0)}};

    const auto box = db::create<db::AddSimpleTags<
        domain::Tags::Faces<2, domain::Tags::Coordinates<2, Frame::Inertial>>,
        domain::Tags::Faces<2, Tags::Beta>,
        domain::Tags::Faces<2, Tags::Gamma>>>(
        std::move(coords_map), std::move(beta_map), std::move(gamma_map));
    elliptic::apply_boundary_condition<
        false, void, standard_boundary_conditions>(
        bc_1, box, direction, make_not_null(&field), make_not_null(&n_dot_grad),
        tnsr::i<ComplexDataVector, 2>{num_pts, 0.0});
    const ComplexDataVector expected_1{
        num_pts, std::complex<double>(0., 0.06226111848298833)};
    CHECK_ITERABLE_APPROX(get(n_dot_grad), expected_1);
  }

  {
    INFO("Testing Order 2");
    const auto created_2 = TestHelpers::test_factory_creation<BC, Sommerfeld>(
        base_sommerfeld + "  HyperboloidalSlicing: True\n  Order: 2");
    const auto& bc_2 = dynamic_cast<const Sommerfeld&>(*created_2);
    auto field = make_with_value<Scalar<ComplexDataVector>>(num_pts, 1.0);
    auto n_dot_grad = make_with_value<Scalar<ComplexDataVector>>(num_pts, 0.0);
    auto beta = make_with_value<
        Scalar<ComplexDataVector>>(num_pts, std::complex<double>(0.5, 0.0));
    auto gamma = make_with_value<tnsr::i<ComplexDataVector, 2>>(num_pts, 0.0);
    get<0>(gamma) = ComplexDataVector{num_pts, {2.0, 0.0}};
    const auto box_2 = db::create<db::AddSimpleTags<
        domain::Tags::Faces<2, domain::Tags::Coordinates<2, Frame::Inertial>>,
        domain::Tags::Faces<2, Tags::Beta>,
        domain::Tags::Faces<2, Tags::Gamma>>>(
            DirectionMap<2, tnsr::I<DataVector, 2>>{
                {direction, tnsr::I<DataVector, 2>{num_pts, 0.0}}},
            DirectionMap<2, Scalar<ComplexDataVector>>{
                {direction, std::move(beta)}},
            DirectionMap<2, tnsr::i<ComplexDataVector, 2>>{
                {direction, std::move(gamma)}});
    elliptic::apply_boundary_condition<
        false, void, standard_boundary_conditions>(
        bc_2, box_2, direction,
        make_not_null(&field), make_not_null(&n_dot_grad),
        tnsr::i<ComplexDataVector, 2>{num_pts, 0.0});
    const ComplexDataVector expected_2(
        num_pts, std::complex<double>(-0.25, 0.0));
    CHECK_ITERABLE_APPROX(get(n_dot_grad), expected_2);
  }
}

}  // namespace ScalarSelfForce::BoundaryConditions
