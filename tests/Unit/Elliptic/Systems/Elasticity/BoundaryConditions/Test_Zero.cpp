// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Elasticity/BoundaryConditions/Zero.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::BoundaryConditions {

SPECTRE_TEST_CASE("Unit.Elasticity.BoundaryConditions.Zero",
                  "[Unit][Elliptic]") {
  // Test factory-creation
  using registrar_fixed =
      Registrars::Zero<2, elliptic::BoundaryConditionType::Dirichlet>;
  using registrar_free =
      Registrars::Zero<2, elliptic::BoundaryConditionType::Neumann>;
  const auto created_fixed = TestHelpers::test_creation<
      std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
          2, tmpl::list<registrar_fixed>>>>("Fixed");
  const auto created_free = TestHelpers::test_creation<
      std::unique_ptr<elliptic::BoundaryConditions::BoundaryCondition<
          2, tmpl::list<registrar_free>>>>("Free");

  // Test semantics
  using Fixed =
      typename registrar_fixed::template f<tmpl::list<registrar_fixed>>;
  using Free = typename registrar_free::template f<tmpl::list<registrar_free>>;
  REQUIRE(dynamic_cast<const Fixed*>(created_fixed.get()) != nullptr);
  REQUIRE(dynamic_cast<const Free*>(created_free.get()) != nullptr);
  const auto& fixed = dynamic_cast<const Fixed&>(*created_fixed);
  const auto& free = dynamic_cast<const Free&>(*created_free);
  test_serialization(fixed);
  test_serialization(free);
  test_copy_semantics(fixed);
  test_copy_semantics(free);
  auto move_fixed = fixed;
  auto move_free = free;
  test_move_semantics(std::move(move_fixed), fixed);
  test_move_semantics(std::move(move_free), free);

  // Test applying the boundary conditions
  const auto box = db::create<db::AddSimpleTags<>>();
  {
    INFO("Dirichlet");
    tnsr::I<DataVector, 2> displacement{3_st,
                                        std::numeric_limits<double>::max()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::max()};
    elliptic::apply_boundary_condition<false>(
        fixed, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress));
    CHECK_ITERABLE_APPROX(get<0>(displacement), SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(displacement), SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Linearized Dirichlet");
    tnsr::I<DataVector, 2> displacement{3_st,
                                        std::numeric_limits<double>::max()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::max()};
    elliptic::apply_boundary_condition<true>(
        fixed, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress));
    CHECK_ITERABLE_APPROX(get<0>(displacement), SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(displacement), SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Neumann");
    tnsr::I<DataVector, 2> displacement{3_st,
                                        std::numeric_limits<double>::max()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::max()};
    elliptic::apply_boundary_condition<false>(
        free, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress));
    CHECK_ITERABLE_APPROX(get<0>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
  }
  {
    INFO("Linearized Neumann");
    tnsr::I<DataVector, 2> displacement{3_st,
                                        std::numeric_limits<double>::max()};
    tnsr::I<DataVector, 2> n_dot_minus_stress{
        3_st, std::numeric_limits<double>::max()};
    elliptic::apply_boundary_condition<true>(
        free, box, Direction<2>::lower_xi(), make_not_null(&displacement),
        make_not_null(&n_dot_minus_stress));
    CHECK_ITERABLE_APPROX(get<0>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
    CHECK_ITERABLE_APPROX(get<1>(n_dot_minus_stress),
                          SINGLE_ARG(DataVector{3, 0.}));
  }
}

}  // namespace Elasticity::BoundaryConditions
