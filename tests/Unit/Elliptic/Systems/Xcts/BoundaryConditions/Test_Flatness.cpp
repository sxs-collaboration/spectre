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
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Flatness.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {
template <size_t EnabledEquations, bool Linearized>
void test_flatness(const Flatness<>& boundary_condition) {
  const size_t num_points = 3;
  const auto box = db::create<db::AddSimpleTags<>>();
  Scalar<DataVector> conformal_factor{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  Scalar<DataVector> n_dot_conformal_factor_gradient{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  if constexpr (EnabledEquations == 1) {
    elliptic::apply_boundary_condition<Linearized>(
        boundary_condition, box, Direction<3>::lower_xi(),
        make_not_null(&conformal_factor),
        make_not_null(&n_dot_conformal_factor_gradient));
  } else {
    Scalar<DataVector> lapse_times_conformal_factor{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    Scalar<DataVector> n_dot_lapse_times_conformal_factor_gradient{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    if constexpr (EnabledEquations == 2) {
      elliptic::apply_boundary_condition<Linearized>(
          boundary_condition, box, Direction<3>::lower_xi(),
          make_not_null(&conformal_factor),
          make_not_null(&lapse_times_conformal_factor),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient));
    } else {
      tnsr::I<DataVector, 3> shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      tnsr::I<DataVector, 3> n_dot_longitudinal_shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      elliptic::apply_boundary_condition<Linearized>(
          boundary_condition, box, Direction<3>::lower_xi(),
          make_not_null(&conformal_factor),
          make_not_null(&lapse_times_conformal_factor),
          make_not_null(&shift_excess),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          make_not_null(&n_dot_longitudinal_shift_excess));
      CHECK(get<0>(shift_excess) == DataVector(num_points, 0.));
      CHECK(get<1>(shift_excess) == DataVector(num_points, 0.));
      CHECK(get<2>(shift_excess) == DataVector(num_points, 0.));
    }
    CHECK(get(lapse_times_conformal_factor) ==
          DataVector(num_points, Linearized ? 0. : 1.));
  }
  CHECK(get(conformal_factor) == DataVector(num_points, Linearized ? 0. : 1.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.Flatness", "[Unit][Elliptic]") {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<
          3, tmpl::list<Registrars::Flatness>>>("Flatness");
  REQUIRE(dynamic_cast<const Flatness<>*>(created.get()) != nullptr);
  const auto& boundary_condition = dynamic_cast<const Flatness<>&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  test_flatness<1, false>(boundary_condition);
  test_flatness<1, true>(boundary_condition);
  test_flatness<2, false>(boundary_condition);
  test_flatness<2, true>(boundary_condition);
  test_flatness<3, false>(boundary_condition);
  test_flatness<3, true>(boundary_condition);
}

}  // namespace Xcts::BoundaryConditions
