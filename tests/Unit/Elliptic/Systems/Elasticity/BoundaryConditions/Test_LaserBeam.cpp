// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Elasticity/HalfSpaceMirror.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

namespace {
template <bool Linearized>
void apply_boundary_condition(
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::I<DataVector, 3>& x, tnsr::i<DataVector, 3> face_normal,
    const double beam_width) {
  const LaserBeam<> laser_beam{beam_width};
  const auto direction = Direction<3>::lower_xi();
  // Normalize the randomly-generated face normal
  const auto face_normal_magnitude = magnitude(face_normal);
  for (size_t d = 0; d < 3; ++d) {
    face_normal.get(d) /= get(face_normal_magnitude);
  }
  const auto box = db::create<db::AddSimpleTags<
      domain::Tags::Faces<3, domain::Tags::Coordinates<3, Frame::Inertial>>,
      domain::Tags::Faces<3, domain::Tags::FaceNormal<3>>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}},
      DirectionMap<3, tnsr::i<DataVector, 3>>{{direction, face_normal}});
  tnsr::I<DataVector, 3> displacement{x.begin()->size(),
                                      std::numeric_limits<double>::max()};
  elliptic::apply_boundary_condition<Linearized>(laser_beam, box, direction,
                                                 make_not_null(&displacement),
                                                 n_dot_minus_stress);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.LaserBeam",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
          3, tmpl::list<Registrars::LaserBeam>>>>(
      "LaserBeam:\n"
      "  BeamWidth: 2.");

  {
    INFO("Semantics");
    REQUIRE(dynamic_cast<const LaserBeam<>*>(created.get()) != nullptr);
    const auto& laser_beam = dynamic_cast<const LaserBeam<>&>(*created);
    test_serialization(laser_beam);
    test_copy_semantics(laser_beam);
    auto move_laser_beam = laser_beam;
    test_move_semantics(std::move(move_laser_beam), laser_beam);
  }

  // Test applying the boundary conditions
  pypp::SetupLocalPythonEnvironment local_python_env(
      "Elliptic/Systems/Elasticity/BoundaryConditions/");
  pypp::check_with_random_values<3>(&apply_boundary_condition<false>,
                                    "LaserBeam", {"normal_dot_minus_stress"},
                                    {{{-2., 2.}, {-1., 1.}, {0.5, 2.}}},
                                    DataVector{3});
  pypp::check_with_random_values<3>(
      &apply_boundary_condition<true>, "LaserBeam",
      {"normal_dot_minus_stress_linearized"},
      {{{-2., 2.}, {-1., 1.}, {0.5, 2.}}}, DataVector{3});

  {
    INFO("Consistency with half-space mirror solution");
    const double beam_width = 2.;
    const DataVector used_for_size{5};
    // Choose an arbitrary set of points on the z=0 surface of the mirror
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-2. * beam_width, 2. * beam_width);
    auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);
    get<2>(x) = 0.;
    // Choose a constitutive relation with arbitrary parameters
    const ConstitutiveRelations::IsotropicHomogeneous<3> constitutive_relation{
        1., 2.};
    // Get the surface stress from the half-space mirror solution
    const Solutions::HalfSpaceMirror<> half_space_mirror{beam_width,
                                                         constitutive_relation};
    const auto minus_stress_solution = get<Tags::MinusStress<3>>(
        half_space_mirror.variables(x, tmpl::list<Tags::MinusStress<3>>{}));
    // Get the stress normal to the surface. The solution assumes the material
    // extends in positive z-direction, so the normal is (0, 0, -1)
    tnsr::I<DataVector, 3> n_dot_minus_stress_solution{used_for_size.size()};
    get<0>(n_dot_minus_stress_solution) = -get<2, 0>(minus_stress_solution);
    get<1>(n_dot_minus_stress_solution) = -get<2, 1>(minus_stress_solution);
    get<2>(n_dot_minus_stress_solution) = -get<2, 2>(minus_stress_solution);
    auto face_normal = make_with_value<tnsr::i<DataVector, 3>>(x, 0.);
    get<2>(face_normal) = -1;
    // Shift the plane where we evaluate the boundary condition along the
    // z-direction, just because it shouldn't affect the result and it might
    // catch issues with computing the coordinate distance
    get<2>(x) += 2.;
    // Compare to the boundary conditions
    const LaserBeam<> laser_beam{beam_width};
    tnsr::I<DataVector, 3> displacement{used_for_size.size(),
                                        std::numeric_limits<double>::max()};
    tnsr::I<DataVector, 3> n_dot_minus_stress{
        used_for_size.size(), std::numeric_limits<double>::max()};
    laser_beam.apply(make_not_null(&displacement),
                     make_not_null(&n_dot_minus_stress), x, face_normal);
    for (size_t d = 0; d < 3; ++d) {
      CAPTURE(d);
      CHECK_ITERABLE_APPROX(n_dot_minus_stress.get(d),
                            n_dot_minus_stress_solution.get(d));
    }
  }
}

}  // namespace Elasticity::BoundaryConditions
