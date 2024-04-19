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
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/Zero.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.Zero",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  using Fixed = Zero<2, elliptic::BoundaryConditionType::Dirichlet>;
  using Free = Zero<2, elliptic::BoundaryConditionType::Neumann>;
  const auto created_fixed = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Fixed>("Fixed");
  const auto created_free = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<2>, Free>("Free");

  // Test semantics
  REQUIRE(dynamic_cast<const Fixed*>(created_fixed.get()) != nullptr);
  REQUIRE(dynamic_cast<const Free*>(created_free.get()) != nullptr);
  const auto& fixed = dynamic_cast<const Fixed&>(*created_fixed);
  const auto& free = dynamic_cast<const Free&>(*created_free);
  {
    INFO("Semantics");
    test_serialization(fixed);
    test_serialization(free);
    test_copy_semantics(fixed);
    test_copy_semantics(free);
    auto move_fixed = fixed;
    auto move_free = free;
    test_move_semantics(std::move(move_fixed), fixed);
    test_move_semantics(std::move(move_free), free);
  }
  {
    INFO("Properties");
    CHECK(fixed.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              2, elliptic::BoundaryConditionType::Dirichlet});
    CHECK(free.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              2, elliptic::BoundaryConditionType::Neumann});
  }

  // Test applying the boundary conditions
  const auto box = db::create<db::AddSimpleTags<>>();
  {
    INFO("Dirichlet");
    tnsr::I<DataVector, 2> displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::iJ<DataVector, 2> deriv_displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    elliptic::apply_boundary_condition<false, void, tmpl::list<Fixed>>(
        fixed, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress), deriv_displacement);
    CHECK_ITERABLE_APPROX(get<0>(displacement), SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(displacement), SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Linearized Dirichlet");
    tnsr::I<DataVector, 2> displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::iJ<DataVector, 2> deriv_displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    elliptic::apply_boundary_condition<true, void, tmpl::list<Fixed>>(
        fixed, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress), deriv_displacement);
    CHECK_ITERABLE_APPROX(get<0>(displacement), SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(displacement), SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Neumann");
    tnsr::I<DataVector, 2> displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::iJ<DataVector, 2> deriv_displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    elliptic::apply_boundary_condition<false, void, tmpl::list<Free>>(
        free, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress), deriv_displacement);
    CHECK_ITERABLE_APPROX(get<0>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Linearized Neumann");
    tnsr::I<DataVector, 2> displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    tnsr::iJ<DataVector, 2> deriv_displacement{
        3_st, std::numeric_limits<double>::signaling_NaN()};
    elliptic::apply_boundary_condition<true, void, tmpl::list<Free>>(
        free, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress), deriv_displacement);
    CHECK_ITERABLE_APPROX(get<0>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
  }
}

}  // namespace Elasticity::BoundaryConditions
