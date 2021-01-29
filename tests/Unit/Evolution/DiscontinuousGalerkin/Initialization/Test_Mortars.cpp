// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {
template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;

  using internal_directions =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;
  using boundary_directions_interior =
      domain::Tags::BoundaryDirectionsInterior<Metavariables::volume_dim>;

  using simple_tags =
      tmpl::list<::Tags::TimeStepId,
                 domain::Tags::Element<Metavariables::volume_dim>,
                 domain::Tags::Mesh<Metavariables::volume_dim>,
                 evolution::dg::Tags::Quadrature>;
  using compute_tags = tmpl::list<
      domain::Tags::InternalDirectionsCompute<Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,

      domain::Tags::BoundaryDirectionsInteriorCompute<
          Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Testing,
                             tmpl::list<Actions::SetupDataBox,
                                        evolution::dg::Initialization::Mortars<
                                            Metavariables::volume_dim>>>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using const_global_cache_tags = tmpl::list<domain::Tags::InitialExtents<Dim>>;

  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, typename MappedType>
using MortarMap =
    std::unordered_map<std::pair<Direction<Dim>, ElementId<Dim>>, MappedType,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>;

template <size_t Dim>
void test_impl(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element, const TimeStepId& time_step_id,
    const Spectral::Quadrature quadrature,
    const MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_meshes,
    const MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>&
        expected_mortar_sizes,
    const DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>&
        expected_normal_covector_quantities) {
  using metavars = Metavariables<Dim>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{initial_extents};
  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, element.id(),
      {time_step_id, element,
       domain::Initialization::create_initial_mesh(
           initial_extents, element.id(), quadrature, {}),
       quadrature});

  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Run the SetupDataBox action
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  element.id());
  // Run the Mortars initialization action
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  element.id());

  const auto get_tag = [&runner, &element](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<component<metavars>, tag>(
        runner, element.id());
  };

  const auto& mortar_meshes = get_tag(Tags::MortarMesh<Dim>{});
  CHECK(mortar_meshes == expected_mortar_meshes);
  const auto& mortar_sizes = get_tag(Tags::MortarSize<Dim>{});
  CHECK(mortar_sizes == expected_mortar_sizes);
  const auto& mortar_data = get_tag(Tags::MortarData<Dim>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    // Just make sure this exists, it is not expected to hold any data
    CHECK(mortar_data.find(mortar_id_and_mesh.first) != mortar_data.end());
  }

  const auto& mortar_next_temporal_ids =
      get_tag(Tags::MortarNextTemporalId<Dim>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    const auto& mortar_id = mortar_id_and_mesh.first;
    if (mortar_id.second != ElementId<Dim>::external_boundary_id()) {
      CHECK(mortar_next_temporal_ids.at(mortar_id) == time_step_id);
    }
  }

  // Cast result of `operator==` to a bool to trick Catch into not trying to
  // stream a nested STL container.
  CHECK(static_cast<bool>(
      get_tag(evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>{}) ==
      expected_normal_covector_quantities));
}

template <size_t Dim>
void test(Spectral::Quadrature quadrature);

template <>
void test<1>(const Spectral::Quadrature quadrature) {
  INFO("1D");
  // Reference element is denoted by X, has one internal boundary and one
  // external boundary:
  //
  // [X| | | ]-> xi

  const ElementId<1> element_id{0, {{{2, 0}}}};
  const ElementId<1> east_id{0, {{{2, 1}}}};
  const std::vector initial_extents{make_array<1>(2_st)};

  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors[Direction<1>::upper_xi()] = Neighbors<1>{{east_id}, {}};
  const Element<1> element{element_id, neighbors};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};

  // We are working with 2 mortars here: a domain boundary at lower xi
  // and an interface at upper xi.
  const auto interface_mortar_id =
      std::make_pair(Direction<1>::upper_xi(), east_id);
  const MortarMap<1, Mesh<0>> expected_mortar_meshes{{interface_mortar_id, {}}};
  const MortarMap<1, std::array<Spectral::MortarSize, 0>> expected_mortar_sizes{
      {interface_mortar_id, {}}};

  const DirectionMap<
      1, std::optional<
             Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                  evolution::dg::Tags::NormalCovector<1>>>>>
      expected_normal_covector_quantities{{Direction<1>::lower_xi(), {}},
                                          {Direction<1>::upper_xi(), {}}};

  test_impl(initial_extents, element, time_step_id, quadrature,
            expected_mortar_meshes, expected_mortar_sizes,
            expected_normal_covector_quantities);
}

template <>
void test<2>(const Spectral::Quadrature quadrature) {
  INFO("2D");
  // Reference element is denoted by X, has two internal boundaries (east and
  // south) and two external boundaries (west and north):
  //
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+

  const ElementId<2> element_id{0, {{{1, 0}, {1, 1}}}};
  const ElementId<2> east_id(0, {{SegmentId{1, 1}, SegmentId{1, 1}}});
  const ElementId<2> south_id(0, {{SegmentId{1, 0}, SegmentId{1, 0}}});
  const std::vector initial_extents{std::array{3_st, 2_st}};

  DirectionMap<2, Neighbors<2>> neighbors{};
  neighbors[Direction<2>::upper_xi()] = Neighbors<2>{{east_id}, {}};
  neighbors[Direction<2>::lower_eta()] = Neighbors<2>{{south_id}, {}};

  const Element<2> element{element_id, neighbors};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};

  // We are working with 4 mortars here: the domain boundary west and north,
  // and interfaces south and east.
  const auto interface_mortar_id_east =
      std::make_pair(Direction<2>::upper_xi(), east_id);
  const auto interface_mortar_id_south =
      std::make_pair(Direction<2>::lower_eta(), south_id);

  const MortarMap<2, Mesh<1>> expected_mortar_meshes{
      {interface_mortar_id_east,
       Mesh<1>(2, Spectral::Basis::Legendre, quadrature)},
      {interface_mortar_id_south,
       Mesh<1>(3, Spectral::Basis::Legendre, quadrature)}};
  MortarMap<2, std::array<Spectral::MortarSize, 1>> expected_mortar_sizes{};
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    expected_mortar_sizes[mortar_id_and_mesh.first] = {
        {Spectral::MortarSize::Full}};
  }

  const DirectionMap<
      2, std::optional<
             Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                  evolution::dg::Tags::NormalCovector<2>>>>>
      expected_normal_covector_quantities{{Direction<2>::lower_xi(), {}},
                                          {Direction<2>::upper_xi(), {}},
                                          {Direction<2>::lower_eta(), {}},
                                          {Direction<2>::upper_eta(), {}}};

  test_impl(initial_extents, element, time_step_id, quadrature,
            expected_mortar_meshes, expected_mortar_sizes,
            expected_normal_covector_quantities);
}

template <>
void test<3>(const Spectral::Quadrature quadrature) {
  INFO("3D");
  // Neighboring elements in:
  // - upper-xi (right id)
  // - lower-eta (front id)
  // - upper-zeta (top id)
  //
  // All other directions don't have neighbors.

  const ElementId<3> element_id{
      0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
  const ElementId<3> right_id(
      0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 0}}});
  const ElementId<3> front_id(
      0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}});
  const ElementId<3> top_id(
      0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 1}}});
  const std::vector initial_extents{std::array{2_st, 3_st, 4_st}};

  DirectionMap<3, Neighbors<3>> neighbors{};
  neighbors[Direction<3>::upper_xi()] = Neighbors<3>{{right_id}, {}};
  neighbors[Direction<3>::lower_eta()] = Neighbors<3>{{front_id}, {}};
  neighbors[Direction<3>::upper_zeta()] = Neighbors<3>{{top_id}, {}};

  const Element<3> element{element_id, neighbors};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};

  const auto interface_mortar_id_right =
      std::make_pair(Direction<3>::upper_xi(), right_id);
  const auto interface_mortar_id_front =
      std::make_pair(Direction<3>::lower_eta(), front_id);
  const auto interface_mortar_id_top =
      std::make_pair(Direction<3>::upper_zeta(), top_id);

  const MortarMap<3, Mesh<2>> expected_mortar_meshes{
      {interface_mortar_id_right,
       Mesh<2>({{3, 4}}, Spectral::Basis::Legendre, quadrature)},
      {interface_mortar_id_front,
       Mesh<2>({{2, 4}}, Spectral::Basis::Legendre, quadrature)},
      {interface_mortar_id_top,
       Mesh<2>({{2, 3}}, Spectral::Basis::Legendre, quadrature)}};
  MortarMap<3, std::array<Spectral::MortarSize, 2>> expected_mortar_sizes{};
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    expected_mortar_sizes[mortar_id_and_mesh.first] = {
        {Spectral::MortarSize::Full, Spectral::MortarSize::Full}};
  }

  const DirectionMap<
      3, std::optional<
             Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                  evolution::dg::Tags::NormalCovector<3>>>>>
      expected_normal_covector_quantities{
          {Direction<3>::lower_xi(), {}},   {Direction<3>::upper_xi(), {}},
          {Direction<3>::lower_eta(), {}},  {Direction<3>::upper_eta(), {}},
          {Direction<3>::lower_zeta(), {}}, {Direction<3>::upper_zeta(), {}}};

  test_impl(initial_extents, element, time_step_id, quadrature,
            expected_mortar_meshes, expected_mortar_sizes,
            expected_normal_covector_quantities);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.Mortars",
                  "[Unit][Evolution]") {
  for (const auto quadrature :
       {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
    test<1>(quadrature);
    test<2>(quadrature);
    test<3>(quadrature);
  }
}
}  // namespace evolution::dg
