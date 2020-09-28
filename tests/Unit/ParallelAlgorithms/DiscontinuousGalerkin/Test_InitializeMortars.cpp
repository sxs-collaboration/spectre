// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct TemporalIdTag : db::SimpleTag {
  using type = int;
};

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct MortarDataTag : db::SimpleTag {
  using type = int;
};

template <size_t Dim, bool Asynchronous>
struct BoundaryScheme {
  static constexpr size_t volume_dim = Dim;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag =
      tmpl::conditional_t<Asynchronous, ::Tags::Next<TemporalIdTag>,
                          temporal_id_tag>;
  using mortar_data_tag = MortarDataTag;
};

template <size_t Dim, typename Metavariables, bool Asynchronous>
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
                                    domain::Tags::InitialExtents<Dim>,
                                    ::Tags::Next<TemporalIdTag>>>,
                     Actions::SetupDataBox, dg::Actions::InitializeDomain<Dim>,
                     Initialization::Actions::AddComputeTags<tmpl::list<
                         domain::Tags::InternalDirectionsCompute<Dim>,
                         domain::Tags::BoundaryDirectionsInteriorCompute<Dim>,
                         domain::Tags::InterfaceCompute<
                             domain::Tags::InternalDirections<Dim>,
                             domain::Tags::Direction<Dim>>,
                         domain::Tags::InterfaceCompute<
                             domain::Tags::BoundaryDirectionsInterior<Dim>,
                             domain::Tags::Direction<Dim>>,
                         domain::Tags::InterfaceCompute<
                             domain::Tags::InternalDirections<Dim>,
                             domain::Tags::InterfaceMesh<Dim>>,
                         domain::Tags::InterfaceCompute<
                             domain::Tags::BoundaryDirectionsInterior<Dim>,
                             domain::Tags::InterfaceMesh<Dim>>>>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              Actions::SetupDataBox,
              dg::Actions::InitializeMortars<BoundaryScheme<Dim, Asynchronous>>,
              // Remove options so that dependencies for
              // `InitializeMortars` are no longer fulfilled in following
              // iterations of the action list. Else `merge_into_databox`
              // would not compile since the added mortar_data_tag is
              // not comparable.
              Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
};

template <size_t Dim, bool Asynchronous>
struct Metavariables {
  using element_array = ElementArray<Dim, Metavariables, Asynchronous>;
  using component_list = tmpl::list<element_array>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, typename GetTag>
void test_next_temporal_ids(
    GetTag&& get_tag,
    const dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_meshes,
    std::true_type /* times_are_asynchronous */) {
  const auto& mortar_next_temporal_ids =
      get_tag(Tags::Mortars<Tags::Next<TemporalIdTag>, Dim>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    const auto& mortar_id = mortar_id_and_mesh.first;
    if (mortar_id.second != ElementId<Dim>::external_boundary_id()) {
      CHECK(mortar_next_temporal_ids.at(mortar_id) == 0);
    }
  }
}

template <size_t Dim, typename GetTag>
void test_next_temporal_ids(
    GetTag&& /* get_tag */,
    const dg::MortarMap<Dim, Mesh<Dim - 1>>& /* expected_mortar_meshes */,
    std::false_type /* times_are_asynchronous */) {}

template <bool Asynchronous, size_t Dim>
void test_initialize_mortars(
    const ElementId<Dim>& element_id, const DomainCreator<Dim>& domain_creator,
    const dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_meshes,
    const dg::MortarMap<Dim, dg::MortarSize<Dim - 1>>& expected_mortar_sizes) {
  CAPTURE(Dim);
  CAPTURE(Asynchronous);
  using metavariables = Metavariables<Dim, Asynchronous>;
  using element_array = typename metavariables::element_array;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), 0});
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
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

  const auto& mortar_meshes =
      get_tag(Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>{});
  CHECK(mortar_meshes == expected_mortar_meshes);
  const auto& mortar_sizes =
      get_tag(Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>{});
  CHECK(mortar_sizes == expected_mortar_sizes);
  const auto& mortar_data = get_tag(::Tags::Mortars<MortarDataTag, Dim>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    // Just make sure this exists, it is not expected to hold any data
    CHECK(mortar_data.find(mortar_id_and_mesh.first) != mortar_data.end());
  }
  test_next_temporal_ids(get_tag, expected_mortar_meshes,
                         std::integral_constant<bool, Asynchronous>{});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelDG.InitializeMortars", "[Unit][Actions]") {
  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    // Reference element:
    // [X| | | ]-> xi
    const ElementId<1> element_id{0, {{{2, 0}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // We are working with 2 mortars here: a domain boundary at lower xi
    // and an interface at upper xi.
    const auto boundary_mortar_id = std::make_pair(
        Direction<1>::lower_xi(), ElementId<1>::external_boundary_id());
    const auto interface_mortar_id = std::make_pair(
        Direction<1>::upper_xi(), ElementId<1>(0, {{SegmentId{2, 1}}}));
    const dg::MortarMap<1, Mesh<0>> expected_mortar_meshes{
        {boundary_mortar_id, {}}, {interface_mortar_id, {}}};
    const dg::MortarMap<1, dg::MortarSize<0>> expected_mortar_sizes{
        {boundary_mortar_id, {}}, {interface_mortar_id, {}}};
    test_initialize_mortars<false>(element_id, domain_creator,
                                   expected_mortar_meshes,
                                   expected_mortar_sizes);
    test_initialize_mortars<true>(element_id, domain_creator,
                                  expected_mortar_meshes,
                                  expected_mortar_sizes);
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
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};
    // We are working with 4 mortars here: the domain boundary west and north,
    // and interfaces south and east.
    const auto boundary_mortar_id_west = std::make_pair(
        Direction<2>::lower_xi(), ElementId<2>::external_boundary_id());
    const auto boundary_mortar_id_north = std::make_pair(
        Direction<2>::upper_eta(), ElementId<2>::external_boundary_id());
    const auto interface_mortar_id_east =
        std::make_pair(Direction<2>::upper_xi(),
                       ElementId<2>(0, {{SegmentId{1, 1}, SegmentId{1, 1}}}));
    const auto interface_mortar_id_south =
        std::make_pair(Direction<2>::lower_eta(),
                       ElementId<2>(0, {{SegmentId{1, 0}, SegmentId{1, 0}}}));
    const dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {boundary_mortar_id_west, Mesh<1>(2, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto)},
        {boundary_mortar_id_north, Mesh<1>(3, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto)},
        {interface_mortar_id_east, Mesh<1>(2, Spectral::Basis::Legendre,
                                           Spectral::Quadrature::GaussLobatto)},
        {interface_mortar_id_south,
         Mesh<1>(3, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto)}};
    dg::MortarMap<2, dg::MortarSize<1>> expected_mortar_sizes{};
    for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
      expected_mortar_sizes[mortar_id_and_mesh.first] = {
          {Spectral::MortarSize::Full}};
    }
    test_initialize_mortars<false>(element_id, domain_creator,
                                   expected_mortar_meshes,
                                   expected_mortar_sizes);
    test_initialize_mortars<true>(element_id, domain_creator,
                                  expected_mortar_meshes,
                                  expected_mortar_sizes);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 1., 3.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{2, 3, 4}}};
    const auto boundary_mortar_id_left = std::make_pair(
        Direction<3>::lower_xi(), ElementId<3>::external_boundary_id());
    const auto boundary_mortar_id_back = std::make_pair(
        Direction<3>::upper_eta(), ElementId<3>::external_boundary_id());
    const auto boundary_mortar_id_bottom = std::make_pair(
        Direction<3>::lower_zeta(), ElementId<3>::external_boundary_id());
    const auto interface_mortar_id_right = std::make_pair(
        Direction<3>::upper_xi(),
        ElementId<3>(0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 0}}}));
    const auto interface_mortar_id_front = std::make_pair(
        Direction<3>::lower_eta(),
        ElementId<3>(0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}}));
    const auto interface_mortar_id_top = std::make_pair(
        Direction<3>::upper_zeta(),
        ElementId<3>(0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 1}}}));
    const dg::MortarMap<3, Mesh<2>> expected_mortar_meshes{
        {boundary_mortar_id_left, Mesh<2>({{3, 4}}, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto)},
        {boundary_mortar_id_back, Mesh<2>({{2, 4}}, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto)},
        {boundary_mortar_id_bottom,
         Mesh<2>({{2, 3}}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto)},
        {interface_mortar_id_right,
         Mesh<2>({{3, 4}}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto)},
        {interface_mortar_id_front,
         Mesh<2>({{2, 4}}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto)},
        {interface_mortar_id_top, Mesh<2>({{2, 3}}, Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto)}};
    dg::MortarMap<3, dg::MortarSize<2>> expected_mortar_sizes{};
    for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
      expected_mortar_sizes[mortar_id_and_mesh.first] = {
          {Spectral::MortarSize::Full, Spectral::MortarSize::Full}};
    }
    test_initialize_mortars<false>(element_id, domain_creator,
                                   expected_mortar_meshes,
                                   expected_mortar_sizes);
    test_initialize_mortars<true>(element_id, domain_creator,
                                  expected_mortar_meshes,
                                  expected_mortar_sizes);
  }
}
