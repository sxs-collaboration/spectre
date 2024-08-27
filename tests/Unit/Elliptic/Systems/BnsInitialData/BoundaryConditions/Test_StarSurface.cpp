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
#include "Elliptic/Systems/BnsInitialData/BoundaryConditions/StarSurface.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/TMPL.hpp"

namespace BnsInitialData::BoundaryConditions {

namespace {
// the velocity potential does not matter
template <bool Linearized>
void apply_star_surface_boundary_condition_specific(
    const gsl::not_null<Scalar<DataVector>*> n_dot_auxilliary_velocity,
    Scalar<DataVector> velocity_potential = Scalar<DataVector>{
        DataVector{1.0, 2.0, 3.0}}) {
  const StarSurface boundary_condition{{}};
  const tnsr::i<DataVector, 3> velocity_potential_gradient{
      {DataVector{1.0, 1.0, 1.0}, DataVector{0.0, 0.0, 0.0},
       DataVector{0.0, 0.0, 0.0}}};
  const Scalar<DataVector> lapse{DataVector{1.0, 1.5, 1.2}};
  const tnsr::I<DataVector, 3> rotational_shift{{DataVector{1.0, 1.0, 1.0},
                                                 DataVector{0.0, 0.0, 0.0},
                                                 DataVector{0.0, 0.0, 0.0}}};
  const auto x = tnsr::I<DataVector, 3>{{DataVector{1.0, 1.0, 1.0},
                                         DataVector{0.0, 0.0, 0.0},
                                         DataVector{0.0, 0.0, 0.0}}};

  const auto direction = Direction<3>::lower_xi();
  const tnsr::i<DataVector, 3> normal{{DataVector{1.0, 0.0, 0.0},
                                       DataVector{0.0, 1.0, 0.0},
                                       DataVector{0.0, 0.0, 1.0}}};
  const double euler_enthalpy_constant = 1.0;
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>,
      domain::Tags::Faces<3, gr::Tags::Lapse<DataVector>>,
      domain::Tags::Faces<3, BnsInitialData::Tags::RotationalShift<DataVector>>,
      BnsInitialData::Tags::EulerEnthalpyConstant>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, normal}},
      DirectionMap<3, Scalar<DataVector>>{{direction, lapse}},
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, rotational_shift}},
      euler_enthalpy_constant);
  elliptic::apply_boundary_condition<Linearized, void, tmpl::list<StarSurface>>(
      boundary_condition, box, Direction<3>::lower_xi(),
      make_not_null(&velocity_potential), n_dot_auxilliary_velocity,
      velocity_potential_gradient);
}

void apply_star_surface_boundary_condition_generic(
    const gsl::not_null<Scalar<DataVector>*> n_dot_auxilliary_velocity,
    Scalar<DataVector> velocity_potential,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& rotational_shift,
    const double euler_enthalpy_constant,
    const tnsr::i<DataVector, 3>& normal) {
  const StarSurface boundary_condition{{}};

  const auto x = tnsr::I<DataVector, 3>{{DataVector{1.0, 1.0, 1.0},
                                         DataVector{0.0, 0.0, 0.0},
                                         DataVector{0.0, 0.0, 0.0}}};

  const auto direction = Direction<3>::lower_xi();

  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>,
      domain::Tags::Faces<3, gr::Tags::Lapse<DataVector>>,
      domain::Tags::Faces<3, BnsInitialData::Tags::RotationalShift<DataVector>>,
      BnsInitialData::Tags::EulerEnthalpyConstant>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, normal}},
      DirectionMap<3, Scalar<DataVector>>{{direction, lapse}},
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, rotational_shift}},
      euler_enthalpy_constant);
  elliptic::apply_boundary_condition<false, void, tmpl::list<StarSurface>>(
      boundary_condition, box, Direction<3>::lower_xi(),
      make_not_null(&velocity_potential), n_dot_auxilliary_velocity,
      velocity_potential_gradient);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.BnsInitialData.BoundaryConditions.StarSurface",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>, StarSurface>(
      "StarSurface:");
  REQUIRE(dynamic_cast<const StarSurface*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const StarSurface&>(*created);
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
        apply_star_surface_boundary_condition_specific<true>(
            make_not_null(&n_dot_auxilliary_velocity_for_linearized));
        CHECK(get(n_dot_auxilliary_velocity_for_linearized) ==
              DataVector{3, 0.0});
      }
      {
        Scalar<DataVector> velocity_potential{DataVector{1.0, 2.0, 3.0}};
        const tnsr::i<DataVector, 3> velocity_potential_gradient{
            {DataVector{1.0, 1.0, 1.0}, DataVector{0.0, 0.0, 0.0},
             DataVector{0.0, 0.0, 0.0}}};
        const Scalar<DataVector> lapse{DataVector{1.0, 1.5, 1.2}};
        const tnsr::I<DataVector, 3> rotational_shift{
            {DataVector{1.0, 1.0, 1.0}, DataVector{0.0, 0.0, 0.0},
             DataVector{0.0, 0.0, 0.0}}};
        const tnsr::i<DataVector, 3> normal{{DataVector{1.0, 0.0, 0.0},
                                             DataVector{0.0, 1.0, 0.0},
                                             DataVector{0.0, 0.0, 1.0}}};
        const double euler_enthalpy_constant = 1.0;
        apply_star_surface_boundary_condition_specific<false>(
            make_not_null(&n_dot_auxilliary_velocity));
        CHECK(get(n_dot_auxilliary_velocity) ==
              euler_enthalpy_constant / square(get(lapse)) *
                  (rotational_shift.get(0) * normal.get(0) +
                   rotational_shift.get(1) * normal.get(1) +
                   rotational_shift.get(2) * normal.get(2)));
      }
    }
    {
      // Compare to python implementation
      pypp::SetupLocalPythonEnvironment local_python_env(
          "Elliptic/Systems/BnsInitialData/BoundaryConditions/");
      pypp::check_with_random_values<6>(
          &apply_star_surface_boundary_condition_generic, "StarSurface",
          {"star_surface_normal_dot_flux"},
          {{{0.0, 1.0},
            {0.0, 1.0},
            {.9, 1.1},
            {-0.5, 0.5},
            {0.0, 0.1},
            {0.0, 1.5}}},
          DataVector{3});
    }
  }
}
}  // namespace BnsInitialData::BoundaryConditions
