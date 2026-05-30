// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace helpers = TestHelpers::domain::BoundaryConditions;

using MockCartoonBC = helpers::TestCartoonBoundaryCondition<3>;
using MockRegularBC = helpers::TestBoundaryCondition<3>;
using MockSystemBCBase = helpers::BoundaryConditionBase<3>;

// Mock metavariables for testing
template <typename BCList>
struct MockMetavariables {
  using system = helpers::SystemWithBoundaryConditions<3>;

  struct factory_creation {
    using factory_classes = tmpl::map<tmpl::pair<MockSystemBCBase, BCList>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Domain.BoundaryConditions.Cartoon.TemplateMetafunctions",
    "[Unit][Domain]") {
  {
    INFO("Testing basic metafunctions");
    // Test inherits_from_mark_as_cartoon trait
    static_assert(
        domain::BoundaryConditions::detail::inherits_from_mark_as_cartoon<
            MockCartoonBC>::value);
    static_assert(
        not domain::BoundaryConditions::detail::inherits_from_mark_as_cartoon<
            MockRegularBC>::value);

    // Test find_cartoon_bc with list containing cartoon BC
    using list_with_cartoon = tmpl::list<MockRegularBC, MockCartoonBC>;
    using found_cartoon =
        domain::BoundaryConditions::detail::find_cartoon_bc<list_with_cartoon>;
    static_assert(std::is_same_v<found_cartoon, MockCartoonBC>);

    // Test find_cartoon_bc with list containing no cartoon BC
    using list_without_cartoon = tmpl::list<MockRegularBC>;
    using found_none = domain::BoundaryConditions::detail::find_cartoon_bc<
        list_without_cartoon>;
    static_assert(std::is_same_v<found_none, void>);

    // Test find_cartoon_bc with empty list
    using empty_list = tmpl::list<>;
    using found_empty =
        domain::BoundaryConditions::detail::find_cartoon_bc<empty_list>;
    static_assert(std::is_same_v<found_empty, void>);

    // Test has_cartoon_bc_v
    static_assert(domain::BoundaryConditions::detail::has_cartoon_bc_v<
                  list_with_cartoon>);
    static_assert(not domain::BoundaryConditions::detail::has_cartoon_bc_v<
                  list_without_cartoon>);
    static_assert(
        not domain::BoundaryConditions::detail::has_cartoon_bc_v<empty_list>);

    // Test filter_out_cartoon_bcs
    using filtered_list =
        domain::BoundaryConditions::detail::filter_out_cartoon_bcs<
            list_with_cartoon>;
    using expected_filtered = tmpl::list<MockRegularBC>;
    static_assert(std::is_same_v<filtered_list, expected_filtered>);

    // Test filter_out_cartoon_bcs with list that has no cartoon BCs
    using filtered_no_cartoon =
        domain::BoundaryConditions::detail::filter_out_cartoon_bcs<
            list_without_cartoon>;
    static_assert(std::is_same_v<filtered_no_cartoon, list_without_cartoon>);

    // Test get_cartoon_boundary_condition_from_system
    using metavars_with_cartoon = MockMetavariables<list_with_cartoon>;
    using system_cartoon_bc =
        domain::BoundaryConditions::get_cartoon_boundary_condition_from_system<
            metavars_with_cartoon>;
    static_assert(std::is_same_v<system_cartoon_bc, MockCartoonBC>);

    using metavars_without_cartoon = MockMetavariables<list_without_cartoon>;
    using system_no_cartoon_bc =
        domain::BoundaryConditions::get_cartoon_boundary_condition_from_system<
            metavars_without_cartoon>;
    static_assert(std::is_same_v<system_no_cartoon_bc, void>);

    // Test get_external_boundary_conditions_from_system
    using external_bcs = domain::BoundaryConditions::
        get_external_boundary_conditions_from_system<metavars_with_cartoon>;
    static_assert(std::is_same_v<external_bcs, expected_filtered>);

    using external_bcs_no_cartoon = domain::BoundaryConditions::
        get_external_boundary_conditions_from_system<metavars_without_cartoon>;
    static_assert(
        std::is_same_v<external_bcs_no_cartoon, list_without_cartoon>);

    // Test system_has_cartoon_bc_v
    static_assert(domain::BoundaryConditions::system_has_cartoon_bc_v<
                  metavars_with_cartoon>);
    static_assert(not domain::BoundaryConditions::system_has_cartoon_bc_v<
                  metavars_without_cartoon>);
  }
  {
    INFO("Testing multiple cartoon BC");
    // Test that having multiple cartoon BCs triggers static_assert
    // We can't test the static_assert directly, but we can test that
    // find_cartoon_bc_impl would catch it at the right place

    // Create a second mock cartoon BC using the existing test helper pattern
    using AnotherMockCartoonBC =
        domain::BoundaryConditions::Cartoon<MockSystemBCBase>;

    // Test that the filter correctly identifies both as cartoon BCs
    using list_with_two_cartoons =
        tmpl::list<MockCartoonBC, AnotherMockCartoonBC, MockRegularBC>;
    using filtered_two_cartoons =
        domain::BoundaryConditions::detail::filter_out_cartoon_bcs<
            list_with_two_cartoons>;
    using expected_filtered_two = tmpl::list<MockRegularBC>;
    static_assert(std::is_same_v<filtered_two_cartoons, expected_filtered_two>);

    // Note: The static_assert in find_cartoon_bc_impl will prevent compilation
    // if someone tries to use get_cartoon_boundary_condition_from_system with
    // a list containing multiple cartoon BCs
  }
  {
    INFO("Testing make cartoon BC");
    // Test make_cartoon_boundary_condition function
    using metavars_with_cartoon =
        MockMetavariables<tmpl::list<MockCartoonBC, MockRegularBC>>;
    using metavars_without_cartoon =
        MockMetavariables<tmpl::list<MockRegularBC>>;

    // Test with system that has cartoon BC
    auto cartoon_bc =
        domain::BoundaryConditions::make_cartoon_boundary_condition<
            metavars_with_cartoon>();
    CHECK(cartoon_bc != nullptr);
    CHECK(domain::BoundaryConditions::is_cartoon(cartoon_bc));

    // Test with system that doesn't have cartoon BC
    auto no_cartoon_bc =
        domain::BoundaryConditions::make_cartoon_boundary_condition<
            metavars_without_cartoon>();
    CHECK(no_cartoon_bc == nullptr);
  }
  {
    INFO("Testing compatibility check");
    CHECK(domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(
        Mesh<3>{{3, 1, 1},
                {Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
                 Spectral::Basis::Cartoon},
                {Spectral::Quadrature::GaussRadauUpper,
                 Spectral::Quadrature::AxialSymmetry,
                 Spectral::Quadrature::AxialSymmetry}}));
    CHECK(domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(Mesh<3>{
        {3, 4, 1},
        {Spectral::Basis::ZernikeB1, Spectral::Basis::Legendre,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::SphericalSymmetry}}));
    CHECK(domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(Mesh<3>{
        {5, 4, 1},
        {Spectral::Basis::Legendre, Spectral::Basis::Chebyshev,
         Spectral::Basis::Cartoon},
        {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
         Spectral::Quadrature::SphericalSymmetry}}));
    CHECK_FALSE(
        domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(Mesh<3>{
            3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}));
    CHECK_FALSE(
        domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(Mesh<2>{
            3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}));
    CHECK_FALSE(
        domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(Mesh<1>{
            3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto}));
    CHECK_FALSE(domain::BoundaryConditions::dg_mesh_is_cartoon_compatible(
        Mesh<3>{{3, 4, 1},
                {Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                 Spectral::Basis::Chebyshev},
                {Spectral::Quadrature::GaussLobatto,
                 Spectral::Quadrature::AxialSymmetry,
                 Spectral::Quadrature::GaussLobatto}}));
  }
}
