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
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/Flatness.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {
template <Xcts::Equations EnabledEquations, bool Linearized>
void test_flatness(const Flatness<EnabledEquations>& boundary_condition) {
  const size_t num_points = 3;
  const auto box = db::create<db::AddSimpleTags<>>();
  Scalar<DataVector> conformal_factor_minus_one{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  Scalar<DataVector> n_dot_conformal_factor_gradient{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  tnsr::i<DataVector, 3> conformal_factor_gradient{
      num_points, std::numeric_limits<double>::signaling_NaN()};
  if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
    elliptic::apply_boundary_condition<Linearized, void,
                                       tmpl::list<Flatness<EnabledEquations>>>(
        boundary_condition, box, Direction<3>::lower_xi(),
        make_not_null(&conformal_factor_minus_one),
        make_not_null(&n_dot_conformal_factor_gradient),
        conformal_factor_gradient);
  } else {
    Scalar<DataVector> lapse_times_conformal_factor_minus_one{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    Scalar<DataVector> n_dot_lapse_times_conformal_factor_gradient{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    tnsr::i<DataVector, 3> lapse_times_conformal_factor_gradient{
        num_points, std::numeric_limits<double>::signaling_NaN()};
    if constexpr (EnabledEquations == Xcts::Equations::HamiltonianAndLapse) {
      elliptic::apply_boundary_condition<
          Linearized, void, tmpl::list<Flatness<EnabledEquations>>>(
          boundary_condition, box, Direction<3>::lower_xi(),
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          conformal_factor_gradient, lapse_times_conformal_factor_gradient);
    } else {
      tnsr::I<DataVector, 3> shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      tnsr::I<DataVector, 3> n_dot_longitudinal_shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      tnsr::iJ<DataVector, 3> deriv_shift_excess{
          num_points, std::numeric_limits<double>::signaling_NaN()};
      elliptic::apply_boundary_condition<
          Linearized, void, tmpl::list<Flatness<EnabledEquations>>>(
          boundary_condition, box, Direction<3>::lower_xi(),
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&shift_excess),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          make_not_null(&n_dot_longitudinal_shift_excess),
          conformal_factor_gradient, lapse_times_conformal_factor_gradient,
          deriv_shift_excess);
      CHECK(get<0>(shift_excess) == DataVector(num_points, 0.));
      CHECK(get<1>(shift_excess) == DataVector(num_points, 0.));
      CHECK(get<2>(shift_excess) == DataVector(num_points, 0.));
    }
    CHECK(get(lapse_times_conformal_factor_minus_one) ==
          DataVector(num_points, 0.));
  }
  CHECK(get(conformal_factor_minus_one) == DataVector(num_points, 0.));
}

template <Xcts::Equations EnabledEquations>
void test_suite() {
  // Test factory-creation
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>,
      Flatness<EnabledEquations>>("Flatness");
  REQUIRE(dynamic_cast<const Flatness<EnabledEquations>*>(created.get()) !=
          nullptr);
  const auto& boundary_condition =
      dynamic_cast<const Flatness<EnabledEquations>&>(*created);
  {
    INFO("Semantics");
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
    auto move_boundary_condition = boundary_condition;
    test_move_semantics(std::move(move_boundary_condition), boundary_condition);
  }
  {
    INFO("Properties");
    if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                1, elliptic::BoundaryConditionType::Dirichlet});
    } else if constexpr (EnabledEquations ==
                         Xcts::Equations::HamiltonianAndLapse) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                2, elliptic::BoundaryConditionType::Dirichlet});
    } else if constexpr (EnabledEquations ==
                         Xcts::Equations::HamiltonianLapseAndShift) {
      CHECK(boundary_condition.boundary_condition_types() ==
            std::vector<elliptic::BoundaryConditionType>{
                5, elliptic::BoundaryConditionType::Dirichlet});
    }
  }
  test_flatness<EnabledEquations, false>(boundary_condition);
  test_flatness<EnabledEquations, true>(boundary_condition);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.Flatness", "[Unit][Elliptic]") {
  test_suite<Xcts::Equations::Hamiltonian>();
  test_suite<Xcts::Equations::HamiltonianAndLapse>();
  test_suite<Xcts::Equations::HamiltonianLapseAndShift>();
}

}  // namespace Xcts::BoundaryConditions
