// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
              tmpl::list<domain::Tags::InitialRefinementLevels<Dim>,
                         domain::Tags::InitialExtents<Dim>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Actions::SetupDataBox, dg::Actions::InitializeDomain<Dim>,
                     // Remove options so that dependencies for
                     // `InitializeDomain` are no longer fulfilled in following
                     // iterations of the action list. Else `merge_into_databox`
                     // would not compile since the added `Tags::ElementMap` is
                     // not comparable.
                     Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim>
void check_compute_items(
    const ActionTesting::MockRuntimeSystem<Metavariables<Dim>>& runner,
    const ElementId<Dim>& element_id) {
  // The compute items themselves are tested elsewhere, so just check if they
  // were indeed added by the initializer
  const auto tag_is_retrievable = [&runner,
                                   &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<
        ElementArray<Dim, Metavariables<Dim>>, tag>(runner, element_id);
  };
  CHECK(tag_is_retrievable(domain::Tags::Coordinates<Dim, Frame::Logical>{}));
  CHECK(tag_is_retrievable(domain::Tags::Coordinates<Dim, Frame::Inertial>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>{}));
  CHECK(tag_is_retrievable(
      domain::Tags::MinimumGridSpacing<Dim, Frame::Inertial>{}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelDG.InitializeDomain", "[Unit][Actions]") {
  {
    INFO("1D");
    // Reference element:
    // [ |X| | ]-> xi
    const ElementId<1> element_id{0, {{SegmentId{2, 1}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{2}}, {{4}}, {{false}}, nullptr};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {domain_creator.create_domain()}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    CHECK(get_tag(domain::Tags::Mesh<1>{}) ==
          Mesh<1>{4, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(get_tag(domain::Tags::Element<1>{}) ==
          Element<1>{element_id,
                     {{Direction<1>::lower_xi(),
                       {{{ElementId<1>{0, {{SegmentId{2, 0}}}}}}, {}}},
                      {Direction<1>::upper_xi(),
                       {{{ElementId<1>{0, {{SegmentId{2, 2}}}}}}, {}}}}});
    const auto& element_map = get_tag(domain::Tags::ElementMap<1>{});
    const tnsr::I<DataVector, 1, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 1, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    const auto& logical_coords =
        get_tag(domain::Tags::Coordinates<1, Frame::Logical>{});
    CHECK(get<0>(logical_coords) ==
          Spectral::collocation_points<Spectral::Basis::Legendre,
                                       Spectral::Quadrature::GaussLobatto>(4));
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<1, Frame::Inertial>{});
    CHECK(inertial_coords == element_map(logical_coords));
    const auto& inverse_jacobian = get_tag(
        domain::Tags::InverseJacobian<1, Frame::Logical, Frame::Inertial>{});
    CHECK(inverse_jacobian == element_map.inv_jacobian(logical_coords));
    const auto& det_inv_jacobian = get_tag(
        domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>{});
    CHECK(det_inv_jacobian == determinant(inverse_jacobian));
  }
  {
    INFO("2D");
    // Reference element:
    // ^ eta
    // +-+-+-+-+> xi
    // | |X| | |
    // +-+-+-+-+
    const ElementId<2> element_id{0, {{SegmentId{2, 1}, SegmentId{0, 0}}}};
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 2.}}, {{false, false}}, {{2, 0}}, {{4, 3}}};
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
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    CHECK(get_tag(domain::Tags::Mesh<2>{}) ==
          Mesh<2>{{{4, 3}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(
        get_tag(domain::Tags::Element<2>{}) ==
        Element<2>{
            element_id,
            {{Direction<2>::lower_xi(),
              {{{ElementId<2>{0, {{SegmentId{2, 0}, SegmentId{0, 0}}}}}}, {}}},
             {Direction<2>::upper_xi(),
              {{{ElementId<2>{0, {{SegmentId{2, 2}, SegmentId{0, 0}}}}}},
               {}}}}});
    const auto& element_map = get_tag(domain::Tags::ElementMap<2>{});
    const tnsr::I<DataVector, 2, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.}, {-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 2, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5}, {0., 0.5, 1., 1.1, 2.}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(inertial_coords_from_element_map),
                          get<1>(expected_inertial_coords));
    const auto& logical_coords =
        get_tag(domain::Tags::Coordinates<2, Frame::Logical>{});
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<2, Frame::Inertial>{});
    CHECK(inertial_coords == element_map(logical_coords));
    const auto& inverse_jacobian = get_tag(
        domain::Tags::InverseJacobian<2, Frame::Logical, Frame::Inertial>{});
    CHECK(inverse_jacobian == element_map.inv_jacobian(logical_coords));
    const auto& det_inv_jacobian = get_tag(
        domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>{});
    CHECK(det_inv_jacobian == determinant(inverse_jacobian));
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 1}}}};
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 2., 3.}},
                                                 {{2, 0, 1}},
                                                 {{4, 3, 2}},
                                                 {{false, false, false}}};
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
        &runner, element_id, {domain_creator.initial_refinement_levels(),
                              domain_creator.initial_extents()});
    ActionTesting::set_phase(make_not_null(&runner),
                             metavariables::Phase::Testing);
    for (size_t i = 0; i < 2; ++i) {
      ActionTesting::next_action<element_array>(make_not_null(&runner),
                                                element_id);
    }
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    check_compute_items(runner, element_id);

    CHECK(get_tag(domain::Tags::Mesh<3>{}) ==
          Mesh<3>{{{4, 3, 2}},
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto});
    CHECK(
        get_tag(domain::Tags::Element<3>{}) ==
        Element<3>{
            element_id,
            {{Direction<3>::lower_xi(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 0}, SegmentId{0, 0}, SegmentId{1, 1}}}}}},
               {}}},
             {Direction<3>::upper_xi(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 2}, SegmentId{0, 0}, SegmentId{1, 1}}}}}},
               {}}},
             {Direction<3>::lower_zeta(),
              {{{ElementId<3>{
                   0, {{SegmentId{2, 1}, SegmentId{0, 0}, SegmentId{1, 0}}}}}},
               {}}}}});
    const auto& element_map = get_tag(domain::Tags::ElementMap<3>{});
    const tnsr::I<DataVector, 3, Frame::Logical> logical_coords_for_element_map{
        {{{-1., -0.5, 0., 0.1, 1.},
          {-1., -0.5, 0., 0.1, 1.},
          {-1., -0.5, 0., 0.1, 1.}}}};
    const auto inertial_coords_from_element_map =
        element_map(logical_coords_for_element_map);
    const tnsr::I<DataVector, 3, Frame::Logical> expected_inertial_coords{
        {{{0., 0.125, 0.25, 0.275, 0.5},
          {0., 0.5, 1., 1.1, 2.},
          {1., 1.5, 2., 2.1, 3.}}}};
    CHECK_ITERABLE_APPROX(get<0>(inertial_coords_from_element_map),
                          get<0>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(inertial_coords_from_element_map),
                          get<1>(expected_inertial_coords));
    CHECK_ITERABLE_APPROX(get<2>(inertial_coords_from_element_map),
                          get<2>(expected_inertial_coords));
    const auto& logical_coords =
        get_tag(domain::Tags::Coordinates<3, Frame::Logical>{});
    const auto& inertial_coords =
        get_tag(domain::Tags::Coordinates<3, Frame::Inertial>{});
    CHECK(inertial_coords == element_map(logical_coords));
    const auto& inverse_jacobian = get_tag(
        domain::Tags::InverseJacobian<3, Frame::Logical, Frame::Inertial>{});
    CHECK(inverse_jacobian == element_map.inv_jacobian(logical_coords));
    const auto& det_inv_jacobian = get_tag(
        domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>{});
    CHECK(det_inv_jacobian == determinant(inverse_jacobian));
  }
}
