// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tags =
      tmpl::list<::Tags::Domain<Dim, Frame::Inertial>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<
                  tmpl::list<::Tags::InitialExtents<Dim>>>,
              dg::Actions::InitializeDomain<Dim>,
              Initialization::Actions::AddComputeTags<tmpl::list<
                  ::Tags::InternalDirections<Dim>,
                  ::Tags::BoundaryDirectionsInterior<Dim>,
                  ::Tags::InterfaceCompute<::Tags::InternalDirections<Dim>,
                                           ::Tags::Direction<Dim>>,
                  ::Tags::InterfaceCompute<
                      ::Tags::BoundaryDirectionsInterior<Dim>,
                      ::Tags::Direction<Dim>>,
                  ::Tags::InterfaceCompute<::Tags::InternalDirections<Dim>,
                                           ::Tags::InterfaceMesh<Dim>>,
                  ::Tags::InterfaceCompute<
                      ::Tags::BoundaryDirectionsInterior<Dim>,
                      ::Tags::InterfaceMesh<Dim>>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::InitializeFluxes<metavariables>>>>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = ::Tags::Variables<tmpl::list<ScalarFieldTag>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.InitializeFluxes",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [X| | | ]-> xi
    const ElementId<1> element_id{0, {{{2, 0}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test that the normal fluxes on the faces have been initialized
    using normal_dot_fluxes_tag =
        db::add_tag_prefix<Tags::NormalDotFlux,
                           typename System<1>::variables_tag>;
    const auto& boundary_normal_dot_fluxes =
        get_tag(Tags::Interface<Tags::BoundaryDirectionsInterior<1>,
                                normal_dot_fluxes_tag>{});
    CHECK(boundary_normal_dot_fluxes.at(Direction<1>::lower_xi())
              .number_of_grid_points() == 1);
    const auto& interface_normal_dot_fluxes = get_tag(
        Tags::Interface<Tags::InternalDirections<1>, normal_dot_fluxes_tag>{});
    CHECK(interface_normal_dot_fluxes.at(Direction<1>::upper_xi())
              .number_of_grid_points() == 1);
  }
  {
    INFO("2D");
    // Reference element:
    // ^ eta
    // +-+-+> xi
    // |X| |
    // +-+-+
    // | | |
    // +-+-+
    const ElementId<2> element_id{0, {{{1, 0}, {1, 1}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test that the normal fluxes on the faces have been initialized
    using normal_dot_fluxes_tag =
        db::add_tag_prefix<Tags::NormalDotFlux,
                           typename System<2>::variables_tag>;
    const auto& boundary_normal_dot_fluxes =
        get_tag(Tags::Interface<Tags::BoundaryDirectionsInterior<2>,
                                normal_dot_fluxes_tag>{});
    CHECK(boundary_normal_dot_fluxes.at(Direction<2>::lower_xi())
              .number_of_grid_points() == 2);
    CHECK(boundary_normal_dot_fluxes.at(Direction<2>::upper_eta())
              .number_of_grid_points() == 3);
    const auto& interface_normal_dot_fluxes = get_tag(
        Tags::Interface<Tags::InternalDirections<2>, normal_dot_fluxes_tag>{});
    CHECK(interface_normal_dot_fluxes.at(Direction<2>::upper_xi())
              .number_of_grid_points() == 2);
    CHECK(interface_normal_dot_fluxes.at(Direction<2>::lower_eta())
              .number_of_grid_points() == 3);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 1., 3.}},
        {{false, false, false}},
        {{1, 1, 1}},
        {{2, 3, 4}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    // Test that the normal fluxes on the faces have been initialized
    using normal_dot_fluxes_tag =
        db::add_tag_prefix<Tags::NormalDotFlux,
                           typename System<2>::variables_tag>;
    const auto& boundary_normal_dot_fluxes =
        get_tag(Tags::Interface<Tags::BoundaryDirectionsInterior<3>,
                                normal_dot_fluxes_tag>{});
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::lower_xi())
              .number_of_grid_points() == 12);
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::upper_eta())
              .number_of_grid_points() == 8);
    CHECK(boundary_normal_dot_fluxes.at(Direction<3>::lower_zeta())
              .number_of_grid_points() == 6);
    const auto& interface_normal_dot_fluxes = get_tag(
        Tags::Interface<Tags::InternalDirections<3>, normal_dot_fluxes_tag>{});
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::upper_xi())
              .number_of_grid_points() == 12);
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::lower_eta())
              .number_of_grid_points() == 8);
    CHECK(interface_normal_dot_fluxes.at(Direction<3>::upper_zeta())
              .number_of_grid_points() == 6);
  }
}
