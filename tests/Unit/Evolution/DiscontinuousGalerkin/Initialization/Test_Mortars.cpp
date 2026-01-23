// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/InterfaceDataPolicy.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarDataHolder.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
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
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 domain::Tags::Element<Metavariables::volume_dim>,
                 domain::Tags::Mesh<Metavariables::volume_dim>,
                 domain::Tags::NeighborMesh<Metavariables::volume_dim>>;
  using compute_tags = tmpl::list<>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<evolution::dg::Initialization::Mortars<
              Metavariables::volume_dim, typename Metavariables::system>>>>;
};

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim, bool LocalTimeStepping>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  struct system {
    using variables_tag = ::Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  };

  using component_list = tmpl::list<component<Metavariables>>;
};

template <size_t Dim>
using dt_variables_tag =
    typename db::add_tag_prefix<::Tags::dt,
                                ::Tags::Variables<tmpl::list<Var1, Var2<Dim>>>>;

template <size_t Dim>
using mortar_data_history_type = typename Tags::MortarDataHistory<
    Dim, typename dt_variables_tag<Dim>::type>::type;

template <bool LocalTimeStepping, size_t Dim>
void test_impl(
    const std::vector<std::array<size_t, Dim>>& initial_extents,
    const Element<Dim>& element, const TimeStepId& time_step_id,
    const TimeStepId& next_time_step_id, const Spectral::Quadrature quadrature,
    const DirectionalIdMap<Dim, Mesh<Dim>>& neighbor_mesh,
    const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_meshes,
    const ::dg::MortarMap<Dim, MortarInfo<Dim>>& expected_mortar_infos,
    const DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>&
        expected_normal_covector_quantities,
    std::optional<Domain<Dim>> domain = std::nullopt) {
  using metavars = Metavariables<Dim, LocalTimeStepping>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  if (domain == std::nullopt) {
    std::vector<Block<Dim>> blocks{};
    blocks.reserve(initial_extents.size());
    for (size_t block_id = 0; block_id < initial_extents.size(); ++block_id) {
      blocks.emplace_back(nullptr, block_id,
                          DirectionMap<Dim, BlockNeighbors<Dim>>{});
    }
    domain = Domain<Dim>{std::move(blocks)};
  }
  tuples::TaggedTuple<domain::Tags::Domain<Dim>> opts{
      std::move(domain.value())};
  MockRuntimeSystem runner{std::move(opts)};
  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, element.id(),
      {time_step_id, next_time_step_id, element,
       domain::create_initial_mesh(initial_extents, element,
                                   Spectral::Basis::Legendre, quadrature),
       neighbor_mesh});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

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
  const auto& mortar_infos = get_tag(Tags::MortarInfo<Dim>{});
  CHECK(mortar_infos == expected_mortar_infos);
  const auto& mortar_data = get_tag(Tags::MortarData<Dim>{});
  const auto& boundary_data_history = get_tag(
      Tags::MortarDataHistory<
          Dim,
          typename db::add_tag_prefix<
              ::Tags::dt, typename metavars::system::variables_tag>::type>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    // Just make sure this exists, it is not expected to hold any data
    CHECK(mortar_data.find(mortar_id_and_mesh.first) != mortar_data.end());
    if (LocalTimeStepping) {
      CHECK(boundary_data_history.find(mortar_id_and_mesh.first) !=
            boundary_data_history.end());
    }
  }

  const auto& mortar_next_temporal_ids =
      get_tag(Tags::MortarNextTemporalId<Dim>{});
  for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
    const auto& mortar_id = mortar_id_and_mesh.first;
    if (mortar_id.id() != ElementId<Dim>::external_boundary_id()) {
      CHECK(mortar_next_temporal_ids.at(mortar_id) == next_time_step_id);
    }
  }

  CHECK(get_tag(evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>{}) ==
        expected_normal_covector_quantities);
}

template <size_t Dim, bool LocalTimeStepping>
struct Test;

template <bool LocalTimeStepping>
struct Test<1, LocalTimeStepping> {
  static void apply(const Spectral::Quadrature quadrature) {
    INFO("1D");
    // Reference element is denoted by X, has one internal boundary and one
    // external boundary:
    //
    // [X| | | ]-> xi

    const ElementId<1> element_id{0, {{{2, 0}}}};
    const ElementId<1> east_id{0, {{{2, 1}}}};
    const std::vector initial_extents{make_array<1>(2_st)};

    // We are working with 2 mortars here: a domain boundary at lower xi
    // and an interface at upper xi.
    const DirectionalId<1> interface_mortar_id{Direction<1>::upper_xi(),
                                               east_id};

    DirectionMap<1, Neighbors<1>> neighbors{};
    DirectionalIdMap<1, Mesh<1>> neighbor_meshes{};
    neighbors[Direction<1>::upper_xi()] =
        Neighbors<1>{{east_id}, OrientationMap<1>::create_aligned()};
    neighbor_meshes[interface_mortar_id] =
        Mesh<1>{initial_extents[0], Spectral::Basis::Legendre, quadrature};
    const Element<1> element{element_id, neighbors};
    const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    const ::dg::MortarMap<1, Mesh<0>> expected_mortar_meshes{
        {interface_mortar_id, {}}};
    const ::dg::MortarMap<1, MortarInfo<1>> expected_mortar_infos{
        {interface_mortar_id,
         MortarInfo<1>{
             {.interface_data_policy =
                  evolution::dg::InterfaceDataPolicy::CopyProject,
              .time_stepping_policy =
                  LocalTimeStepping
                      ? evolution::dg::TimeSteppingPolicy::Conservative
                      : evolution::dg::TimeSteppingPolicy::EqualRate}}}};

    const DirectionMap<
        1, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<1>>>>>
        expected_normal_covector_quantities{{Direction<1>::lower_xi(), {}},
                                            {Direction<1>::upper_xi(), {}}};

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature, neighbor_meshes,
                                 expected_mortar_meshes, expected_mortar_infos,
                                 expected_normal_covector_quantities);
  }
};

template <bool LocalTimeStepping>
struct Test<2, LocalTimeStepping> {
  static void apply(const Spectral::Quadrature quadrature) {
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

    // We are working with 4 mortars here: the domain boundary west and north,
    // and interfaces south and east.
    const DirectionalId<2> interface_mortar_id_east{Direction<2>::upper_xi(),
                                                    east_id};
    const DirectionalId<2> interface_mortar_id_south{Direction<2>::lower_eta(),
                                                     south_id};

    DirectionMap<2, Neighbors<2>> neighbors{};
    neighbors[Direction<2>::upper_xi()] =
        Neighbors<2>{{east_id}, OrientationMap<2>::create_aligned()};
    neighbors[Direction<2>::lower_eta()] =
        Neighbors<2>{{south_id}, OrientationMap<2>::create_aligned()};
    DirectionalIdMap<2, Mesh<2>> neighbor_meshes{};
    neighbor_meshes[interface_mortar_id_east] =
        Mesh<2>{initial_extents[0], Spectral::Basis::Legendre, quadrature};
    neighbor_meshes[interface_mortar_id_south] =
        Mesh<2>{initial_extents[0], Spectral::Basis::Legendre, quadrature};

    const Element<2> element{element_id, neighbors};
    const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {interface_mortar_id_east,
         Mesh<1>(2, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_south,
         Mesh<1>(3, Spectral::Basis::Legendre, quadrature)}};
    ::dg::MortarMap<2, MortarInfo<2>> expected_mortar_infos{};
    for (const auto& mortar_id_and_mesh : expected_mortar_meshes) {
      expected_mortar_infos.emplace(
          mortar_id_and_mesh.first,
          MortarInfo<2>{
              {.mortar_size = {{Spectral::SegmentSize::Full}},
               .interface_data_policy =
                   evolution::dg::InterfaceDataPolicy::CopyProject,
               .time_stepping_policy =
                   LocalTimeStepping
                       ? evolution::dg::TimeSteppingPolicy::Conservative
                       : evolution::dg::TimeSteppingPolicy::EqualRate}});
    }

    const DirectionMap<
        2, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<2>>>>>
        expected_normal_covector_quantities{{Direction<2>::lower_xi(), {}},
                                            {Direction<2>::upper_xi(), {}},
                                            {Direction<2>::lower_eta(), {}},
                                            {Direction<2>::upper_eta(), {}}};

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature, neighbor_meshes,
                                 expected_mortar_meshes, expected_mortar_infos,
                                 expected_normal_covector_quantities);
  }
};

template <bool LocalTimeStepping>
struct Test<3, LocalTimeStepping> {
  static void apply(const Spectral::Quadrature quadrature) {
    INFO("3D");
    // Neighboring elements in:
    // - upper-xi (right id)
    // - lower-eta (front id)
    // - upper-zeta (top id)
    //
    // All other directions don't have neighbors.

    const ElementId<3> element_id{
        0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const ElementId<3> right_id(
        1, {{SegmentId{2, 1}, SegmentId{1, 0}, SegmentId{1, 1}}});
    const ElementId<3> front_id(
        0, {{SegmentId{1, 1}, SegmentId{1, 0}, SegmentId{1, 0}}});
    const ElementId<3> top_id(
        0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 1}}});
    const std::vector initial_extents{std::array{2_st, 3_st, 4_st},
                                      std::array{5_st, 6_st, 7_st}};

    const DirectionalId<3> interface_mortar_id_right{Direction<3>::upper_xi(),
                                                     right_id};
    const DirectionalId<3> interface_mortar_id_front{Direction<3>::lower_eta(),
                                                     front_id};
    const DirectionalId<3> interface_mortar_id_top{Direction<3>::upper_zeta(),
                                                   top_id};

    DirectionMap<3, Neighbors<3>> neighbors{};
    neighbors[Direction<3>::upper_xi()] =
        Neighbors<3>{{right_id},
                     OrientationMap<3>{{Direction<3>::upper_eta(),
                                        Direction<3>::upper_zeta(),
                                        Direction<3>::upper_xi()}}};
    neighbors[Direction<3>::lower_eta()] =
        Neighbors<3>{{front_id}, OrientationMap<3>::create_aligned()};
    neighbors[Direction<3>::upper_zeta()] =
        Neighbors<3>{{top_id}, OrientationMap<3>::create_aligned()};
    DirectionalIdMap<3, Mesh<3>> neighbor_meshes{};
    neighbor_meshes[interface_mortar_id_right] =
        Mesh<3>{{{6_st, 7_st, 5_st}}, Spectral::Basis::Legendre, quadrature};
    neighbor_meshes[interface_mortar_id_front] =
        Mesh<3>{initial_extents[0], Spectral::Basis::Legendre, quadrature};
    neighbor_meshes[interface_mortar_id_top] =
        Mesh<3>{initial_extents[0], Spectral::Basis::Legendre, quadrature};

    const Element<3> element{element_id, neighbors};
    const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    const ::dg::MortarMap<3, Mesh<2>> expected_mortar_meshes{
        {interface_mortar_id_right,
         Mesh<2>({{7, 5}}, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_front,
         Mesh<2>({{2, 4}}, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_top,
         Mesh<2>({{2, 3}}, Spectral::Basis::Legendre, quadrature)}};
    const auto expected_time_stepping_policy =
        LocalTimeStepping ? evolution::dg::TimeSteppingPolicy::Conservative
                          : evolution::dg::TimeSteppingPolicy::EqualRate;
    ::dg::MortarMap<3, MortarInfo<3>> expected_mortar_infos{};
    expected_mortar_infos.emplace(
        interface_mortar_id_right,
        MortarInfo<3>{
            {.mortar_size = {{Spectral::SegmentSize::Full,
                              Spectral::SegmentSize::UpperHalf}},
             .interface_data_policy =
                 evolution::dg::InterfaceDataPolicy::OrientCopyProject,
             .time_stepping_policy = expected_time_stepping_policy}});
    expected_mortar_infos.emplace(
        interface_mortar_id_front,
        MortarInfo<3>{{.mortar_size = {{Spectral::SegmentSize::Full,
                                        Spectral::SegmentSize::Full}},
                       .interface_data_policy =
                           evolution::dg::InterfaceDataPolicy::CopyProject,
                       .time_stepping_policy = expected_time_stepping_policy}});
    expected_mortar_infos.emplace(
        interface_mortar_id_top,
        MortarInfo<3>{{.mortar_size = {{Spectral::SegmentSize::Full,
                                        Spectral::SegmentSize::Full}},
                       .interface_data_policy =
                           evolution::dg::InterfaceDataPolicy::CopyProject,
                       .time_stepping_policy = expected_time_stepping_policy}});

    const DirectionMap<
        3, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<3>>>>>
        expected_normal_covector_quantities{
            {Direction<3>::lower_xi(), {}},   {Direction<3>::upper_xi(), {}},
            {Direction<3>::lower_eta(), {}},  {Direction<3>::upper_eta(), {}},
            {Direction<3>::lower_zeta(), {}}, {Direction<3>::upper_zeta(), {}}};

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature, neighbor_meshes,
                                 expected_mortar_meshes, expected_mortar_infos,
                                 expected_normal_covector_quantities);
  }
};

template <bool LocalTimeStepping>
void test_nonconforming_blocks() {
  INFO("NonconformingSphericalShells");
  const auto creator = domain::creators::NonconformingSphericalShells(
      2.0, 3.0, 4.0, 0, 0, 5, 7, 11, nullptr, nullptr);
  auto domain = creator.create_domain();
  const auto initial_refinement = creator.initial_refinement_levels();
  const auto initial_extents = creator.initial_extents();
  const ElementId<3> shell_id{6};
  const Element<3> shell = domain::create_initial_element(
      shell_id, domain.blocks(), initial_refinement);
  const Mesh<3> shell_mesh = domain::create_initial_mesh(
      initial_extents, shell, Spectral::Basis::Legendre,
      Spectral::Quadrature::GaussLobatto);
  const Mesh<2> shell_mortar_mesh = shell_mesh.slice_away(0_st);
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};
  const auto expected_time_stepping_policy =
      LocalTimeStepping ? evolution::dg::TimeSteppingPolicy::Conservative
                        : evolution::dg::TimeSteppingPolicy::EqualRate;
  {
    INFO("Test S2 shell");
    const auto& shell_neighbor_ids =
        shell.neighbors().at(Direction<3>::lower_xi());
    DirectionalIdMap<3, Mesh<3>> shell_neighbor_meshes{};
    for (const auto id : shell_neighbor_ids) {
      const DirectionalId<3> shell_neighbor_directional_id{
          Direction<3>::lower_xi(), id};
      const Element<3> neighbor_element = domain::create_initial_element(
          id, domain.blocks(), initial_refinement);
      const Mesh<3> neighbor_mesh = domain::create_initial_mesh(
          initial_extents, neighbor_element, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto);
      shell_neighbor_meshes[shell_neighbor_directional_id] = neighbor_mesh;
    }
    const DirectionalId<3> shell_mortar_id{Direction<3>::lower_xi(), shell_id};

    ::dg::MortarMap<3, Mesh<2>> shell_expected_mortar_meshes{};
    shell_expected_mortar_meshes[shell_mortar_id] = shell_mortar_mesh;
    ::dg::MortarMap<3, MortarInfo<3>> shell_expected_mortar_infos{};
    shell_expected_mortar_infos.emplace(
        shell_mortar_id,
        MortarInfo<3>{{.interface_data_policy =
                           evolution::dg::InterfaceDataPolicy::
                               NonconformingNeighborInterpolates,
                       .time_stepping_policy = expected_time_stepping_policy}});
    const DirectionMap<
        3, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<3>>>>>
        shell_expected_normal_covector_quantities{
            {Direction<3>::lower_xi(), {}}, {Direction<3>::upper_xi(), {}}};
    test_impl<LocalTimeStepping>(
        initial_extents, shell, time_step_id, next_time_step_id,
        Spectral::Quadrature::GaussLobatto, shell_neighbor_meshes,
        shell_expected_mortar_meshes, shell_expected_mortar_infos,
        shell_expected_normal_covector_quantities,
        std::make_optional<Domain<3>>(std::move(domain)));
  }
  {
    INFO("Test cubed sphere");
    domain = creator.create_domain();
    const ElementId<3> element_id{2};
    const Element<3> element = domain::create_initial_element(
        element_id, domain.blocks(), initial_refinement);
    const Mesh<3> volume_mesh = domain::create_initial_mesh(
        initial_extents, element, Spectral::Basis::Legendre,
        Spectral::Quadrature::GaussLobatto);
    DirectionalIdMap<3, Mesh<3>> neighbor_meshes{};
    ::dg::MortarMap<3, Mesh<2>> expected_mortar_meshes{};
    ::dg::MortarMap<3, MortarInfo<3>> expected_mortar_infos{};
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const DirectionalId<3> mortar_id{direction, neighbor};
        const auto& neighbor_block = domain.blocks()[neighbor.block_id()];
        const Element<3> neighbor_element = domain::create_initial_element(
            neighbor, domain.blocks(), initial_refinement);
        if (neighbors.are_conforming()) {
          const auto& neighbor_orientation = neighbors.orientation(neighbor);
          neighbor_meshes.emplace(
              mortar_id,
              neighbor_orientation.inverse_map()(::domain::create_initial_mesh(
                  initial_extents, neighbor_block, neighbor,
                  Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto)));
        } else {
          neighbor_meshes.emplace(mortar_id,
                                  ::domain::create_initial_mesh(
                                      initial_extents, neighbor_block, neighbor,
                                      Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto));
        }
        const Mesh<2> face_mesh = volume_mesh.slice_away(direction.dimension());
        expected_mortar_meshes[mortar_id] = face_mesh;
        const auto& neighbor_orientation = neighbors.orientation(neighbor);
        if (direction == Direction<3>::upper_zeta()) {
          expected_mortar_infos.emplace(
              mortar_id,
              MortarInfo<3>{
                  {.interpolator =
                       ::dg::MortarInterpolator<3>{element_id, mortar_id,
                                                   domain, face_mesh,
                                                   shell_mortar_mesh},
                   .interface_data_policy = evolution::dg::InterfaceDataPolicy::
                       NonconformingSelfInterpolates,
                   .time_stepping_policy = expected_time_stepping_policy}});
        } else {
          expected_mortar_infos.emplace(
              mortar_id,
              MortarInfo<3>{
                  {.mortar_size = {{Spectral::SegmentSize::Full,
                                    Spectral::SegmentSize::Full}},
                   .interface_data_policy =
                       neighbor_orientation.is_aligned()
                           ? InterfaceDataPolicy::CopyProject
                           : InterfaceDataPolicy::OrientCopyProject,
                   .time_stepping_policy = expected_time_stepping_policy}});
        }
      }
    }
    const DirectionMap<
        3, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<3>>>>>
        expected_normal_covector_quantities{
            {Direction<3>::lower_xi(), {}},   {Direction<3>::upper_xi(), {}},
            {Direction<3>::lower_eta(), {}},  {Direction<3>::upper_eta(), {}},
            {Direction<3>::lower_zeta(), {}}, {Direction<3>::upper_zeta(), {}}};
    test_impl<LocalTimeStepping>(
        initial_extents, element, time_step_id, next_time_step_id,
        Spectral::Quadrature::GaussLobatto, neighbor_meshes,
        expected_mortar_meshes, expected_mortar_infos,
        expected_normal_covector_quantities,
        std::make_optional<Domain<3>>(std::move(domain)));
  }
}

template <size_t Dim>
void check_mortar_data(const MortarData<Dim>& projected,
                       const MortarData<Dim>& expected) {
  CHECK(projected.mortar_mesh == expected.mortar_mesh);
  CHECK(projected.face_mesh == expected.face_mesh);
  CHECK(projected.volume_mesh == expected.volume_mesh);
  if (projected.mortar_data.has_value()) {
    CHECK_ITERABLE_APPROX(projected.mortar_data.value(),
                          expected.mortar_data.value());
  } else {
    CHECK_FALSE(expected.mortar_data.has_value());
  }
  if (projected.face_normal_magnitude.has_value()) {
    CHECK_ITERABLE_APPROX(projected.face_normal_magnitude.value(),
                          expected.face_normal_magnitude.value());
  } else {
    CHECK_FALSE(expected.face_normal_magnitude.has_value());
  }
  if (projected.face_det_jacobian.has_value()) {
    CHECK_ITERABLE_APPROX(projected.face_det_jacobian.value(),
                          expected.face_det_jacobian.value());
  } else {
    CHECK_FALSE(expected.face_det_jacobian.has_value());
  }
  if (projected.volume_det_inv_jacobian.has_value()) {
    CHECK_ITERABLE_APPROX(projected.volume_det_inv_jacobian.value(),
                          expected.volume_det_inv_jacobian.value());
  } else {
    CHECK_FALSE(expected.volume_det_inv_jacobian.has_value());
  }
}

template <size_t Dim, typename CouplingResult>
void check_boundary_histories(
    const ::dg::MortarMap<
        Dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<Dim>,
                                           ::evolution::dg::MortarData<Dim>,
                                           CouplingResult>>& value,
    const ::dg::MortarMap<
        Dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<Dim>,
                                           ::evolution::dg::MortarData<Dim>,
                                           CouplingResult>>& expected) {
  using HistMap = std::decay_t<decltype(value)>;
  const auto compare_entries = [](const HistMap& a, const HistMap& b) {
    for (const auto& [mortar, history_a] : a) {
      CAPTURE(mortar);
      const auto it = b.find(mortar);
      REQUIRE(it != b.end());
      const auto& history_b = it->second;
      const auto local_a = history_a.local();
      const auto local_b = history_b.local();
      local_a.for_each([&](const TimeStepId& id,
                           const ::evolution::dg::MortarData<Dim>& data) {
        CHECK(local_a.integration_order(id) == local_b.integration_order(id));
        check_mortar_data(data, local_b.data(id));
      });
      const auto remote_a = history_a.remote();
      const auto remote_b = history_b.remote();
      remote_a.for_each([&](const TimeStepId& id,
                            const ::evolution::dg::MortarData<Dim>& data) {
        CHECK(remote_a.integration_order(id) == remote_b.integration_order(id));
        check_mortar_data(data, remote_b.data(id));
      });
    }
  };
  compare_entries(value, expected);
  compare_entries(expected, value);
}

template <size_t Dim, bool UsingLts>
void test_p_refine(
    ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data,
    ::dg::MortarMap<Dim, Mesh<Dim - 1>> mortar_mesh,
    ::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos,
    ::dg::MortarMap<Dim, TimeStepId> mortar_next_temporal_id,
    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>
        normal_covector_and_magnitude,
    mortar_data_history_type<Dim> mortar_data_history,
    const Mesh<Dim>& old_mesh, Mesh<Dim> new_mesh,
    const Element<Dim>& old_element, Element<Dim> new_element,
    ::dg::MortarMap<Dim, Mesh<Dim>> neighbor_meshes,
    const TimeStepId& temporal_id,
    const ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>&
        expected_mortar_data,
    const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_mesh,
    const ::dg::MortarMap<Dim, MortarInfo<Dim>>& expected_mortar_infos,
    const ::dg::MortarMap<Dim, TimeStepId>& expected_mortar_next_temporal_id,
    const DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>&
        expected_normal_covector_and_magnitude,
    const mortar_data_history_type<Dim>& expected_mortar_data_history) {
  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Domain<Dim>, domain::Tags::Mesh<Dim>,
      domain::Tags::Element<Dim>, domain::Tags::NeighborMesh<Dim>,
      ::Tags::TimeStepId, Tags::MortarData<Dim>, Tags::MortarMesh<Dim>,
      Tags::MortarInfo<Dim>, Tags::MortarNextTemporalId<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      Tags::MortarDataHistory<Dim, typename dt_variables_tag<Dim>::type>>>(
      Domain<Dim>{}, std::move(new_mesh), std::move(new_element),
      std::move(neighbor_meshes), temporal_id, std::move(mortar_data),
      std::move(mortar_mesh), std::move(mortar_infos),
      std::move(mortar_next_temporal_id),
      std::move(normal_covector_and_magnitude), std::move(mortar_data_history));

  db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
      Metavariables<Dim, UsingLts>>>(make_not_null(&box),
                                     std::make_pair(old_mesh, old_element));

  CHECK(db::get<Tags::MortarData<Dim>>(box) == expected_mortar_data);
  CHECK(db::get<Tags::MortarMesh<Dim>>(box) == expected_mortar_mesh);
  CHECK(db::get<Tags::MortarInfo<Dim>>(box) == expected_mortar_infos);
  CHECK(db::get<Tags::MortarNextTemporalId<Dim>>(box) ==
        expected_mortar_next_temporal_id);
  CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box) ==
        expected_normal_covector_and_magnitude);
  check_boundary_histories(
      db::get<
          Tags::MortarDataHistory<Dim, typename dt_variables_tag<Dim>::type>>(
          box),
      expected_mortar_data_history);
}

template <size_t Dim>
Element<Dim> make_element();

template <>
Element<1> make_element<1>() {
  const ElementId<1> element_id{0, {{SegmentId{2, 0}}}};
  const ElementId<1> neighbor_id{0, {{SegmentId{2, 1}}}};
  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors[Direction<1>::upper_xi()] =
      Neighbors<1>{{neighbor_id}, OrientationMap<1>::create_aligned()};
  return Element<1>{element_id, neighbors};
}

template <>
Element<2> make_element<2>() {
  const ElementId<2> element_id{0, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
  const ElementId<2> east_id(0, {{SegmentId{1, 1}, SegmentId{1, 1}}});
  const ElementId<2> south_id(0, {{SegmentId{1, 0}, SegmentId{1, 0}}});
  DirectionMap<2, Neighbors<2>> neighbors{};
  neighbors[Direction<2>::upper_xi()] =
      Neighbors<2>{{east_id}, OrientationMap<2>::create_aligned()};
  neighbors[Direction<2>::lower_eta()] =
      Neighbors<2>{{south_id}, OrientationMap<2>::create_aligned()};
  return Element<2>{element_id, neighbors};
}

template <>
Element<3> make_element<3>() {
  const ElementId<3> element_id{
      0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
  const ElementId<3> right_id(
      0, {{SegmentId{1, 1}, SegmentId{1, 1}, SegmentId{1, 0}}});
  const ElementId<3> front_id(
      0, {{SegmentId{1, 0}, SegmentId{1, 0}, SegmentId{1, 0}}});
  const ElementId<3> top_id(
      0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 1}}});
  DirectionMap<3, Neighbors<3>> neighbors{};
  neighbors[Direction<3>::upper_xi()] =
      Neighbors<3>{{right_id}, OrientationMap<3>::create_aligned()};
  neighbors[Direction<3>::lower_eta()] =
      Neighbors<3>{{front_id}, OrientationMap<3>::create_aligned()};
  neighbors[Direction<3>::upper_zeta()] =
      Neighbors<3>{{top_id}, OrientationMap<3>::create_aligned()};
  return Element<3>{element_id, neighbors};
}

template <size_t Dim>
void test_p_refine_gts() {
  const Mesh<Dim> old_mesh{2, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  Mesh<Dim> new_mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  Mesh<Dim> neighbor_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  const auto old_element = make_element<Dim>();
  auto new_element = make_element<Dim>();
  const TimeStepId next_temporal_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> mortar_mesh{};
  ::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_and_magnitude{};
  mortar_data_history_type<Dim> mortar_data_history{};

  ::dg::MortarMap<Dim, TimeStepId> mortar_next_temporal_ids{};
  ::dg::MortarMap<Dim, Mesh<Dim>> neighbor_meshes{};
  for (const auto& [direction, neighbors] : old_element.neighbors()) {
    normal_covector_and_magnitude[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
      mortar_mesh.emplace(
          mortar_id,
          ::dg::mortar_mesh(old_mesh.slice_away(direction.dimension()),
                            neighbor_mesh.slice_away(direction.dimension())));
      const auto& neighbor_orientation = neighbors.orientation(neighbor);
      mortar_infos.emplace(
          mortar_id,
          MortarInfo<Dim>{
              {.mortar_size = ::dg::mortar_size(old_element.id(), neighbor,
                                                direction.dimension(),
                                                neighbor_orientation),
               .interface_data_policy =
                   neighbor_orientation.is_aligned()
                       ? InterfaceDataPolicy::CopyProject
                       : InterfaceDataPolicy::OrientCopyProject,
               .time_stepping_policy = TimeSteppingPolicy::EqualRate}});
      mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
      neighbor_meshes.emplace(mortar_id, neighbor_mesh);
    }
  }

  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>
      expected_mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> expected_mortar_mesh{};
  ::dg::MortarMap<Dim, MortarInfo<Dim>> expected_mortar_infos{};
  ::dg::MortarMap<Dim, TimeStepId> expected_mortar_next_temporal_ids{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      expected_normal_covector_and_magnitude{};
  mortar_data_history_type<Dim> expected_mortar_data_history{};
  for (const auto& [direction, neighbors] : new_element.neighbors()) {
    expected_normal_covector_and_magnitude[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      expected_mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
      expected_mortar_mesh.emplace(
          mortar_id,
          ::dg::mortar_mesh(new_mesh.slice_away(direction.dimension()),
                            neighbor_mesh.slice_away(direction.dimension())));
      const auto& neighbor_orientation = neighbors.orientation(neighbor);
      expected_mortar_infos.emplace(
          mortar_id,
          MortarInfo<Dim>{
              {.mortar_size = ::dg::mortar_size(new_element.id(), neighbor,
                                                direction.dimension(),
                                                neighbor_orientation),
               .interface_data_policy =
                   neighbor_orientation.is_aligned()
                       ? evolution::dg::InterfaceDataPolicy::CopyProject
                       : evolution::dg::InterfaceDataPolicy::OrientCopyProject,
               .time_stepping_policy = TimeSteppingPolicy::EqualRate}});
      expected_mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
    }
  }
  for (const auto& direction : new_element.external_boundaries()) {
    normal_covector_and_magnitude[direction] = std::nullopt;
    expected_normal_covector_and_magnitude[direction] = std::nullopt;
  }

  test_p_refine<Dim, false>(
      std::move(mortar_data), std::move(mortar_mesh), std::move(mortar_infos),
      std::move(mortar_next_temporal_ids),
      std::move(normal_covector_and_magnitude), std::move(mortar_data_history),
      old_mesh, std::move(new_mesh), old_element, std::move(new_element),
      neighbor_meshes, next_temporal_id, expected_mortar_data,
      expected_mortar_mesh, expected_mortar_infos,
      expected_mortar_next_temporal_ids, expected_normal_covector_and_magnitude,
      expected_mortar_data_history);
}

// The data arrays are set to linear functions, with element_size and
// mortar_size are used to choose the domain the function is evaluated
// on.  The mortar_size is relative to the size of the element, and at
// least one of the two must be Full in each dimension.
template <size_t Dim>
MortarData<Dim> make_mortar_data(
    const Mesh<Dim - 1>& mortar_mesh, const Mesh<Dim - 1>& face_mesh,
    const Mesh<Dim>& volume_mesh, const bool is_local_side, const double value,
    const std::array<Spectral::SegmentSize, Dim>& element_size,
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size,
    const size_t dimension) {
  const auto linear_func =
      []<size_t D>(const Mesh<D>& mesh,
                   const std::array<Spectral::SegmentSize, D>& size) {
        if constexpr (D == 0) {
          return DataVector{1.0};
        } else {
          const auto coords = logical_coordinates(mesh);
          DataVector linear(mesh.number_of_grid_points(), 0.0);
          for (size_t i = 0; i < D; ++i) {
            switch (gsl::at(size, i)) {
              case Spectral::SegmentSize::LowerHalf:
                linear += 0.5 * (coords.get(i) - 1.0);
                break;
              case Spectral::SegmentSize::UpperHalf:
                linear += 0.5 * (coords.get(i) + 1.0);
                break;
              default:
                ASSERT(gsl::at(size, i) == Spectral::SegmentSize::Full,
                       "Bad argument: " << gsl::at(size, i));
                linear += coords.get(i);
                break;
            }
          }
          return linear;
        }
      };

  const auto face_size = all_but_specified_element_of(element_size, dimension);
  auto absolute_mortar_size = mortar_size;
  for (size_t i = 0; i < Dim - 1; ++i) {
    if (gsl::at(face_size, i) != Spectral::SegmentSize::Full) {
      ASSERT(gsl::at(mortar_size, i) == Spectral::SegmentSize::Full,
             "Can't represent a quarter segment.");
      gsl::at(absolute_mortar_size, i) = gsl::at(face_size, i);
    }
  }

  MortarData<Dim> mortar_data;
  mortar_data.mortar_data.emplace(
      value * linear_func(mortar_mesh, absolute_mortar_size));
  mortar_data.mortar_mesh.emplace(mortar_mesh);
  if (is_local_side) {
    mortar_data.face_normal_magnitude.emplace(
        2.0 * value * linear_func(face_mesh, face_size));
    mortar_data.face_det_jacobian.emplace(3.0 * value *
                                          linear_func(face_mesh, face_size));
    mortar_data.face_mesh.emplace(face_mesh);
    mortar_data.volume_det_inv_jacobian.emplace(
        4.0 * value * linear_func(volume_mesh, element_size));
    mortar_data.volume_mesh.emplace(volume_mesh);
  }
  return mortar_data;
}

template <size_t Dim>
MortarData<Dim> make_mortar_data(const Mesh<Dim - 1>& mortar_mesh,
                                 const Mesh<Dim - 1>& face_mesh,
                                 const Mesh<Dim>& volume_mesh,
                                 const bool is_local_side, const double value) {
  return make_mortar_data<Dim>(
      mortar_mesh, face_mesh, volume_mesh, is_local_side, value,
      make_array<Dim>(Spectral::SegmentSize::Full),
      make_array<Dim - 1>(Spectral::SegmentSize::Full),
      // Dimension doesn't matter for full-size elements
      0);
}

template <size_t Dim>
using boundary_history_type =
    typename mortar_data_history_type<Dim>::mapped_type;

template <size_t Dim>
void test_p_refine_lts() {
  const Mesh<Dim> old_mesh{4, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  Mesh<Dim> new_mesh{5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> neighbor_mesh{3, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};

  const auto old_element = make_element<Dim>();
  auto new_element = make_element<Dim>();
  const TimeStepId next_temporal_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};
  const std::vector<TimeStepId> local_past_ids{
      TimeStepId{true, 3, Time{Slab{0.2, 3.4}, {0, 100}}},
      TimeStepId{true, 3, Time{Slab{0.2, 3.4}, {2, 100}}},
      TimeStepId{true, 3, Time{Slab{0.2, 3.4}, {4, 100}}}};
  const std::vector<TimeStepId> remote_past_ids{
      TimeStepId{true, 3, Time{Slab{0.2, 3.4}, {0, 100}}},
      TimeStepId{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}}};

  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>> mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> mortar_mesh{};
  ::dg::MortarMap<Dim, MortarInfo<Dim>> mortar_infos{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_and_magnitude{};
  mortar_data_history_type<Dim> mortar_data_history{};
  ::dg::MortarMap<Dim, TimeStepId> mortar_next_temporal_ids{};
  ::dg::MortarMap<Dim, Mesh<Dim>> neighbor_meshes{};
  for (const auto& [direction, neighbors] : old_element.neighbors()) {
    normal_covector_and_magnitude[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
      mortar_mesh.emplace(
          mortar_id,
          ::dg::mortar_mesh(old_mesh.slice_away(direction.dimension()),
                            neighbor_mesh.slice_away(direction.dimension())));
      const auto& neighbor_orientation = neighbors.orientation(neighbor);
      mortar_infos.emplace(
          mortar_id,
          MortarInfo<Dim>{
              {.mortar_size = ::dg::mortar_size(old_element.id(), neighbor,
                                                direction.dimension(),
                                                neighbor_orientation),
               .interface_data_policy =
                   neighbor_orientation.is_aligned()
                       ? InterfaceDataPolicy::CopyProject
                       : InterfaceDataPolicy::OrientCopyProject,
               .time_stepping_policy = TimeSteppingPolicy::Conservative}});
      mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
      neighbor_meshes.emplace(mortar_id, neighbor_mesh);
      mortar_data_history.emplace(mortar_id, boundary_history_type<Dim>{});
      for (size_t i = 0; i < 3; ++i) {
        mortar_data_history.at(mortar_id).local().insert(
            local_past_ids[i], 3,
            make_mortar_data(mortar_mesh.at(mortar_id),
                             old_mesh.slice_away(direction.dimension()),
                             old_mesh, true, 5.0 + static_cast<double>(i)));
      }
      for (size_t i = 0; i < 2; ++i) {
        mortar_data_history.at(mortar_id).remote().insert(
            local_past_ids[i], 3,
            make_mortar_data(mortar_mesh.at(mortar_id),
                             neighbor_mesh.slice_away(direction.dimension()),
                             neighbor_mesh, false,
                             5.0 + static_cast<double>(i)));
      }
    }
  }

  ::dg::MortarMap<Dim, evolution::dg::MortarDataHolder<Dim>>
      expected_mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> expected_mortar_mesh{};
  ::dg::MortarMap<Dim, MortarInfo<Dim>> expected_mortar_infos{};
  ::dg::MortarMap<Dim, TimeStepId> expected_mortar_next_temporal_ids{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      expected_normal_covector_and_magnitude{};
  mortar_data_history_type<Dim> expected_mortar_data_history{};
  for (const auto& [direction, neighbors] : new_element.neighbors()) {
    expected_normal_covector_and_magnitude[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      expected_mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
      expected_mortar_mesh.emplace(
          mortar_id,
          ::dg::mortar_mesh(new_mesh.slice_away(direction.dimension()),
                            neighbor_mesh.slice_away(direction.dimension())));
      const auto& neighbor_orientation = neighbors.orientation(neighbor);
      expected_mortar_infos.emplace(
          mortar_id,
          MortarInfo<Dim>{
              {.mortar_size = ::dg::mortar_size(new_element.id(), neighbor,
                                                direction.dimension(),
                                                neighbor_orientation),
               .interface_data_policy =
                   neighbor_orientation.is_aligned()
                       ? InterfaceDataPolicy::CopyProject
                       : InterfaceDataPolicy::OrientCopyProject,
               .time_stepping_policy = TimeSteppingPolicy::Conservative}});
      expected_mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
      expected_mortar_data_history.emplace(mortar_id,
                                           boundary_history_type<Dim>{});
      // These use the old mortar_mesh, not expected_mortar_mesh,
      // because mortar data is not projected during p-refinement.
      for (size_t i = 0; i < 3; ++i) {
        expected_mortar_data_history.at(mortar_id).local().insert(
            local_past_ids[i], 3,
            make_mortar_data(mortar_mesh.at(mortar_id),
                             new_mesh.slice_away(direction.dimension()),
                             new_mesh, true, 5.0 + static_cast<double>(i)));
      }
      for (size_t i = 0; i < 2; ++i) {
        expected_mortar_data_history.at(mortar_id).remote().insert(
            local_past_ids[i], 3,
            make_mortar_data(mortar_mesh.at(mortar_id),
                             neighbor_mesh.slice_away(direction.dimension()),
                             neighbor_mesh, false,
                             5.0 + static_cast<double>(i)));
      }
    }
  }
  for (const auto& direction : new_element.external_boundaries()) {
    normal_covector_and_magnitude[direction] = std::nullopt;
    expected_normal_covector_and_magnitude[direction] = std::nullopt;
  }

  test_p_refine<Dim, true>(
      std::move(mortar_data), std::move(mortar_mesh), std::move(mortar_infos),
      std::move(mortar_next_temporal_ids),
      std::move(normal_covector_and_magnitude), std::move(mortar_data_history),
      old_mesh, std::move(new_mesh), old_element, std::move(new_element),
      neighbor_meshes, next_temporal_id, expected_mortar_data,
      expected_mortar_mesh, expected_mortar_infos,
      expected_mortar_next_temporal_ids, expected_normal_covector_and_magnitude,
      expected_mortar_data_history);
}

template <size_t Dim>
Mesh<Dim> lgl_mesh(const size_t uniform_extents) {
  return {uniform_extents, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

template <size_t Dim>
Mesh<Dim> lgl_mesh(const std::array<size_t, Dim>& extents) {
  return {extents, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

template <size_t NumMortars>
::dg::MortarMap<2, evolution::dg::MortarDataHolder<2>> empty_mortar_data(
    const std::array<DirectionalId<2>, NumMortars>& mortar_ids) {
  ::dg::MortarMap<2, evolution::dg::MortarDataHolder<2>> mortar_data{};
  for (const auto& mortar_id : mortar_ids) {
    mortar_data.emplace(mortar_id, evolution::dg::MortarDataHolder<2>{});
  }
  return mortar_data;
}

template <size_t NumMortars>
::dg::MortarMap<2, TimeStepId> constant_next_temporal_ids(
    const std::array<DirectionalId<2>, NumMortars>& mortar_ids,
    const TimeStepId& temporal_id) {
  ::dg::MortarMap<2, TimeStepId> next_temporal_ids{};
  for (const auto& mortar_id : mortar_ids) {
    next_temporal_ids.emplace(mortar_id, temporal_id);
  }
  return next_temporal_ids;
}

template <size_t Dim>
boundary_history_type<Dim> make_boundary_history(
    const DirectionalId<Dim>& mortar_id, const Mesh<Dim>& volume_mesh,
    const Mesh<Dim - 1>& mortar_mesh,
    const std::array<Spectral::SegmentSize, Dim>& element_size,
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size,
    const bool include_local, const bool include_remote) {
  const TimeStepId history_temporal_id(true, 4, Slab(2.0, 3.0).start());
  boundary_history_type<Dim> history{};
  if (include_local) {
    const auto face_mesh =
        volume_mesh.slice_away(mortar_id.direction().dimension());
    history.local().insert(
        history_temporal_id, 1,
        make_mortar_data<Dim>(
            mortar_mesh, face_mesh, volume_mesh, true,
            static_cast<double>(mortar_id.id().block_id()) + 1.0, element_size,
            mortar_size, mortar_id.direction().dimension()));
  }
  if (include_remote) {
    history.remote().insert(
        history_temporal_id, 1,
        make_mortar_data<Dim>(
            mortar_mesh, {}, {}, false,
            static_cast<double>(mortar_id.id().block_id()) + 2.0, element_size,
            mortar_size, mortar_id.direction().dimension()));
  }
  return history;
}

template <size_t NumMortars>
Tags::MortarDataHistory<2, dt_variables_tag<2>::type>::type
make_boundary_histories(
    const std::array<DirectionalId<2>, NumMortars>& mortar_ids,
    const Mesh<2>& volume_mesh,
    const ::dg::MortarMap<2, Mesh<1>>& mortar_meshes,
    const std::array<Spectral::SegmentSize, 2>& element_size,
    const ::dg::MortarMap<2, MortarInfo<2>>& mortar_infos,
    const bool include_local, const bool include_remote) {
  Tags::MortarDataHistory<2, dt_variables_tag<2>::type>::type histories{};
  for (const auto& mortar_id : mortar_ids) {
    histories.emplace(
        mortar_id, make_boundary_history(
                       mortar_id, volume_mesh, mortar_meshes.at(mortar_id),
                       element_size, mortar_infos.at(mortar_id).mortar_size(),
                       include_local, include_remote));
  }
  return histories;
}

// For h-refinement tests, we use a 2D element and have neighbors
// change as:
//
//     +---+   eta        +---+
//     |a  |   |          |   |
//     |   |   +- xi      | p-|
//     |   |              |ref|
// +---+---+---+      +-+-+---+---+
// |b  |   |c  |      | |f|   |g  |
// |   | X |   |  =>  | | | X +---+
// |   |   |   |      | | |   |h  |
// +---+-+-+---+      +-+-+---+---+
//     |d|e|              |i  |
//     | | |              |   |
//     | | |              |   |
//     +-+-+              +---+
//
// Originally, a,b,c,d have mesh (3,4) and e has (5,6), all in the
// coordinates of the central element.  After refining, a has (4,5), i
// has (5,6), and the others are still (3,4).
//
// The central region can do nothing, p-refine, split, or join.
template <bool LocalTimeStepping>
void test_h_refinement() {
  using NormalVars =
      Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                           evolution::dg::Tags::NormalCovector<2>>>;
  using mortar_data_history_tag =
      Tags::MortarDataHistory<2, dt_variables_tag<2>::type>;

  const DirectionalId<2> mortar_id_a(Direction<2>::upper_eta(),
                                     ElementId<2>(0));
  const DirectionalId<2> mortar_id_b(Direction<2>::lower_xi(), ElementId<2>(1));
  const DirectionalId<2> mortar_id_c(Direction<2>::upper_xi(), ElementId<2>(3));
  const DirectionalId<2> mortar_id_d(Direction<2>::lower_eta(),
                                     ElementId<2>(4, {{{1, 0}, {0, 0}}}));
  const DirectionalId<2> mortar_id_e(Direction<2>::lower_eta(),
                                     ElementId<2>(4, {{{1, 1}, {0, 0}}}));
  // rotated
  const DirectionalId<2> mortar_id_f(Direction<2>::lower_xi(),
                                     ElementId<2>(1, {{{0, 0}, {1, 1}}}));
  const DirectionalId<2> mortar_id_g(Direction<2>::upper_xi(),
                                     ElementId<2>(3, {{{0, 0}, {1, 1}}}));
  const DirectionalId<2> mortar_id_h(Direction<2>::upper_xi(),
                                     ElementId<2>(3, {{{0, 0}, {1, 0}}}));
  const DirectionalId<2> mortar_id_i(Direction<2>::lower_eta(),
                                     ElementId<2>(4));
  const std::array orig_mortar_ids{mortar_id_a, mortar_id_b, mortar_id_c,
                                   mortar_id_d, mortar_id_e};
  const std::array refined_mortar_ids{mortar_id_a, mortar_id_f, mortar_id_g,
                                      mortar_id_h, mortar_id_i};

  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  const OrientationMap<2> rotated{
      std::array{Direction<2>::upper_eta(), Direction<2>::lower_xi()}};
  const TimeStepId temporal_id(true, 5, Slab(3.0, 6.0).start());
  const Mesh<2> orig_mesh = lgl_mesh<2>({{3, 4}});
  const auto time_stepping_policy = LocalTimeStepping
                                        ? TimeSteppingPolicy::Conservative
                                        : TimeSteppingPolicy::EqualRate;

  tuples::TaggedTuple<domain::Tags::Mesh<2>, domain::Tags::Element<2>,
                      domain::Tags::NeighborMesh<2>, ::Tags::TimeStepId,
                      Tags::MortarData<2>, Tags::MortarMesh<2>,
                      Tags::MortarInfo<2>, Tags::MortarNextTemporalId<2>,
                      evolution::dg::Tags::NormalCovectorAndMagnitude<2>,
                      mortar_data_history_tag>
      orig_single_items{};
  {
    const ElementId<2> id(2);

    get<domain::Tags::Mesh<2>>(orig_single_items) = orig_mesh;

    get<domain::Tags::Element<2>>(orig_single_items) = Element<2>(
        id,
        {{mortar_id_a.direction(), Neighbors<2>({mortar_id_a.id()}, rotated)},
         {mortar_id_b.direction(), Neighbors<2>({mortar_id_b.id()}, rotated)},
         {mortar_id_c.direction(), Neighbors<2>({mortar_id_c.id()}, aligned)},
         {mortar_id_d.direction(),
          Neighbors<2>({mortar_id_d.id(), mortar_id_e.id()}, aligned)}});

    get<domain::Tags::NeighborMesh<2>>(orig_single_items) = {
        {mortar_id_a, orig_mesh},
        {mortar_id_b, orig_mesh},
        {mortar_id_c, orig_mesh},
        {mortar_id_d, orig_mesh},
        {mortar_id_e, lgl_mesh<2>({{5, 6}})}};

    get<::Tags::TimeStepId>(orig_single_items) = temporal_id;

    get<Tags::MortarData<2>>(orig_single_items) =
        empty_mortar_data(orig_mortar_ids);

    get<Tags::MortarMesh<2>>(orig_single_items) = {
        {mortar_id_a, lgl_mesh<1>(3)},
        {mortar_id_b, lgl_mesh<1>(4)},
        {mortar_id_c, lgl_mesh<1>(4)},
        {mortar_id_d, lgl_mesh<1>(3)},
        {mortar_id_e, lgl_mesh<1>(5)}};

    get<Tags::MortarInfo<2>>(orig_single_items) = {
        {mortar_id_a,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_b,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_c,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_d,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::LowerHalf}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_e,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::UpperHalf}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}}};

    get<Tags::MortarNextTemporalId<2>>(orig_single_items) =
        constant_next_temporal_ids(orig_mortar_ids, temporal_id);

    // Values aren't used, except for checking that they haven't changed.
    get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(
        orig_single_items) = {{Direction<2>::upper_eta(), NormalVars(3, 1.0)},
                              {Direction<2>::lower_xi(), NormalVars(4, 2.0)},
                              {Direction<2>::upper_xi(), NormalVars(4, 3.0)},
                              {Direction<2>::lower_eta(), NormalVars(3, 4.0)}};

    if (LocalTimeStepping) {
      get<mortar_data_history_tag>(orig_single_items) = make_boundary_histories(
          orig_mortar_ids, orig_mesh,
          get<Tags::MortarMesh<2>>(orig_single_items),
          make_array<2>(Spectral::SegmentSize::Full),
          get<Tags::MortarInfo<2>>(orig_single_items), true, true);
    }
  }

  const auto refined_a_mesh = lgl_mesh<2>({{4, 5}});

  const Element<2> refined_single_element(
      get<domain::Tags::Element<2>>(orig_single_items).id(),
      {{mortar_id_a.direction(), Neighbors<2>({mortar_id_a.id()}, rotated)},
       {mortar_id_f.direction(), Neighbors<2>({mortar_id_f.id()}, rotated)},
       {mortar_id_g.direction(),
        Neighbors<2>({mortar_id_g.id(), mortar_id_h.id()}, aligned)},
       {mortar_id_i.direction(), Neighbors<2>({mortar_id_i.id()}, aligned)}});

  const auto& orig_neighbor_meshes =
      get<domain::Tags::NeighborMesh<2>>(orig_single_items);
  const ::dg::MortarMap<2, Mesh<2>> refined_single_neighbor_meshes{
      {mortar_id_a, refined_a_mesh},
      {mortar_id_f, orig_neighbor_meshes.at(mortar_id_b)},
      {mortar_id_g, orig_neighbor_meshes.at(mortar_id_c)},
      {mortar_id_h, orig_neighbor_meshes.at(mortar_id_c)},
      {mortar_id_i, orig_neighbor_meshes.at(mortar_id_e)}};

  const ::dg::MortarMap<2, MortarInfo<2>> expected_single_mortar_infos{
      {mortar_id_a,
       MortarInfo<2>{
           {.mortar_size = {{Spectral::SegmentSize::Full}},
            .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
            .time_stepping_policy = time_stepping_policy}}},
      {mortar_id_f,
       MortarInfo<2>{
           {.mortar_size = {{Spectral::SegmentSize::Full}},
            .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
            .time_stepping_policy = time_stepping_policy}}},
      {mortar_id_g,
       MortarInfo<2>{{.mortar_size = {{Spectral::SegmentSize::UpperHalf}},
                      .interface_data_policy = InterfaceDataPolicy::CopyProject,
                      .time_stepping_policy = time_stepping_policy}}},
      {mortar_id_h,
       MortarInfo<2>{{.mortar_size = {{Spectral::SegmentSize::LowerHalf}},
                      .interface_data_policy = InterfaceDataPolicy::CopyProject,
                      .time_stepping_policy = time_stepping_policy}}},
      {mortar_id_i,
       MortarInfo<2>{{.mortar_size = {{Spectral::SegmentSize::Full}},
                      .interface_data_policy = InterfaceDataPolicy::CopyProject,
                      .time_stepping_policy = time_stepping_policy}}}};

  const DirectionMap<2, std::optional<NormalVars>>
      empty_normal_covector_and_magnitude{
          {Direction<2>::upper_eta(), std::nullopt},
          {Direction<2>::lower_xi(), std::nullopt},
          {Direction<2>::upper_xi(), std::nullopt},
          {Direction<2>::lower_eta(), std::nullopt}};

  {
    INFO("No local refinement");
    auto box = tmpl::as_pack<decltype(orig_single_items)>(
        [&]<typename... Tags>(tmpl::type_<Tags>... /*meta*/) {
          return db::create<
              db::AddSimpleTags<domain::Tags::Domain<2>, Tags...>>(
              Domain<2>{}, get<Tags>(orig_single_items)...);
        });

    const auto& mortar_ids = refined_mortar_ids;
    db::mutate<domain::Tags::Element<2>, domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = refined_single_element;
          *neighbor_meshes = refined_single_neighbor_meshes;
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(
        make_not_null(&box),
        std::pair(get<domain::Tags::Mesh<2>>(orig_single_items),
                  get<domain::Tags::Element<2>>(orig_single_items)));

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_a, lgl_mesh<1>(4)},
        {mortar_id_f, lgl_mesh<1>(4)},
        {mortar_id_g, lgl_mesh<1>(4)},
        {mortar_id_h, lgl_mesh<1>(4)},
        {mortar_id_i, lgl_mesh<1>(5)}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      expected_mortar_data_history = make_boundary_histories(
          refined_mortar_ids, orig_mesh, expected_mortar_meshes,
          make_array<2>(Spectral::SegmentSize::Full),
          expected_single_mortar_infos, true, false);
      // No projection when no h-refinement
      expected_mortar_data_history[mortar_id_a] =
          get<mortar_data_history_tag>(orig_single_items).at(mortar_id_a);
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_single_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(
              orig_single_items));
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  {
    INFO("Local p-refinement");
    auto box = tmpl::as_pack<decltype(orig_single_items)>(
        [&]<typename... Tags>(tmpl::type_<Tags>... /*meta*/) {
          return db::create<
              db::AddSimpleTags<domain::Tags::Domain<2>, Tags...>>(
              Domain<2>{}, get<Tags>(orig_single_items)...);
        });

    const auto& mortar_ids = refined_mortar_ids;
    const auto refined_mesh = lgl_mesh<2>({{4, 5}});
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = refined_single_element;
          *mesh = refined_mesh;
          *neighbor_meshes = refined_single_neighbor_meshes;
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(
        make_not_null(&box),
        std::pair(get<domain::Tags::Mesh<2>>(orig_single_items),
                  get<domain::Tags::Element<2>>(orig_single_items)));

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_a, lgl_mesh<1>(4)},
        {mortar_id_f, lgl_mesh<1>(5)},
        {mortar_id_g, lgl_mesh<1>(5)},
        {mortar_id_h, lgl_mesh<1>(5)},
        {mortar_id_i, lgl_mesh<1>(5)}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      expected_mortar_data_history = make_boundary_histories(
          refined_mortar_ids, refined_mesh, expected_mortar_meshes,
          make_array<2>(Spectral::SegmentSize::Full),
          expected_single_mortar_infos, true, false);

      // No projection of mortar data when no h-refinement, but
      // geometric data is projected.
      expected_mortar_data_history[mortar_id_a] = make_boundary_history(
          mortar_id_a, refined_mesh,
          get<Tags::MortarMesh<2>>(orig_single_items).at(mortar_id_a),
          make_array<2>(Spectral::SegmentSize::Full),
          expected_single_mortar_infos.at(mortar_id_a).mortar_size(), true,
          true);
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_single_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  const ElementId<2> id_nw(2, {{{1, 0}, {1, 1}}});
  const ElementId<2> id_ne(2, {{{1, 1}, {1, 1}}});
  const ElementId<2> id_sw(2, {{{1, 0}, {1, 0}}});
  const ElementId<2> id_se(2, {{{1, 1}, {1, 0}}});
  const auto element_size_nw = domain::child_size(
      id_nw.segment_ids(), refined_single_element.id().segment_ids());
  const auto element_size_ne = domain::child_size(
      id_ne.segment_ids(), refined_single_element.id().segment_ids());
  const auto element_size_sw = domain::child_size(
      id_sw.segment_ids(), refined_single_element.id().segment_ids());
  const auto element_size_se = domain::child_size(
      id_se.segment_ids(), refined_single_element.id().segment_ids());
  const DirectionalId<2> mortar_id_nw_ne(Direction<2>::upper_xi(), id_ne);
  const DirectionalId<2> mortar_id_nw_sw(Direction<2>::lower_eta(), id_sw);
  const DirectionalId<2> mortar_id_ne_nw(Direction<2>::lower_xi(), id_nw);
  const DirectionalId<2> mortar_id_ne_se(Direction<2>::lower_eta(), id_se);
  const DirectionalId<2> mortar_id_sw_se(Direction<2>::upper_xi(), id_se);
  const DirectionalId<2> mortar_id_sw_nw(Direction<2>::upper_eta(), id_nw);
  const DirectionalId<2> mortar_id_se_sw(Direction<2>::lower_xi(), id_sw);
  const DirectionalId<2> mortar_id_se_ne(Direction<2>::upper_eta(), id_ne);

  {
    INFO("Join");
    auto box =
        db::create<tmpl::push_front<decltype(orig_single_items)::tags_list,
                                    domain::Tags::Domain<2>>>();

    const auto& mortar_ids = refined_mortar_ids;
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = refined_single_element;
          *mesh = orig_mesh;
          *neighbor_meshes = refined_single_neighbor_meshes;
        },
        make_not_null(&box));

    using ChildItems =
        tuples::TaggedTuple<::Tags::TimeStepId, mortar_data_history_tag>;
    mortar_data_history_tag::type history_nw{};
    mortar_data_history_tag::type history_ne{};
    mortar_data_history_tag::type history_sw{};
    mortar_data_history_tag::type history_se{};
    if (LocalTimeStepping) {
      // Only the mortar_size is used for constructing the histories
      const auto dummy_mortar_infos = [](const auto& ids) {
        ::dg::MortarMap<2, MortarInfo<2>> infos{};
        for (const auto& id : ids) {
          infos.emplace(
              id,
              MortarInfo<2>{{.mortar_size = {{Spectral::SegmentSize::Full}}}});
        }
        return infos;
      };
      const std::array mortar_ids_nw{mortar_id_a, mortar_id_b, mortar_id_nw_ne,
                                     mortar_id_nw_sw};
      const std::array mortar_ids_ne{mortar_id_a, mortar_id_ne_nw, mortar_id_c,
                                     mortar_id_ne_se};
      const std::array mortar_ids_sw{mortar_id_sw_nw, mortar_id_b,
                                     mortar_id_sw_se, mortar_id_d};
      const std::array mortar_ids_se{mortar_id_se_ne, mortar_id_se_sw,
                                     mortar_id_c, mortar_id_e};
      const ::dg::MortarMap<2, Mesh<1>> mortar_meshes_nw{
          {mortar_id_a, lgl_mesh<1>(3)},
          {mortar_id_b, lgl_mesh<1>(4)},
          {mortar_id_nw_ne, lgl_mesh<1>(4)},
          {mortar_id_nw_sw, lgl_mesh<1>(3)}};
      const ::dg::MortarMap<2, Mesh<1>> mortar_meshes_ne{
          {mortar_id_a, lgl_mesh<1>(3)},
          {mortar_id_ne_nw, lgl_mesh<1>(4)},
          {mortar_id_c, lgl_mesh<1>(4)},
          {mortar_id_ne_se, lgl_mesh<1>(3)}};
      const ::dg::MortarMap<2, Mesh<1>> mortar_meshes_sw{
          {mortar_id_sw_nw, lgl_mesh<1>(3)},
          {mortar_id_b, lgl_mesh<1>(4)},
          {mortar_id_sw_se, lgl_mesh<1>(4)},
          {mortar_id_d, lgl_mesh<1>(3)}};
      const ::dg::MortarMap<2, Mesh<1>> mortar_meshes_se{
          {mortar_id_se_ne, lgl_mesh<1>(3)},
          {mortar_id_se_sw, lgl_mesh<1>(4)},
          {mortar_id_c, lgl_mesh<1>(4)},
          {mortar_id_e, lgl_mesh<1>(5)}};
      history_nw = make_boundary_histories(
          mortar_ids_nw, orig_mesh, mortar_meshes_nw, element_size_nw,
          dummy_mortar_infos(mortar_ids_nw), true, true);
      history_ne = make_boundary_histories(
          mortar_ids_ne, orig_mesh, mortar_meshes_ne, element_size_ne,
          dummy_mortar_infos(mortar_ids_ne), true, true);
      history_sw = make_boundary_histories(
          mortar_ids_sw, orig_mesh, mortar_meshes_sw, element_size_sw,
          dummy_mortar_infos(mortar_ids_sw), true, true);
      history_se = make_boundary_histories(
          mortar_ids_se, orig_mesh, mortar_meshes_se, element_size_se,
          dummy_mortar_infos(mortar_ids_se), true, true);
    }
    const std::unordered_map<ElementId<2>, ChildItems> children_items{
        {id_nw, {temporal_id, std::move(history_nw)}},
        {id_ne, {temporal_id, std::move(history_ne)}},
        {id_sw, {temporal_id, std::move(history_sw)}},
        {id_se, {temporal_id, std::move(history_se)}}};

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(make_not_null(&box),
                                              children_items);

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_a, lgl_mesh<1>(4)},
        {mortar_id_f, lgl_mesh<1>(4)},
        {mortar_id_g, lgl_mesh<1>(4)},
        {mortar_id_h, lgl_mesh<1>(4)},
        {mortar_id_i, lgl_mesh<1>(5)}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      for (const auto& mortar_id : refined_mortar_ids) {
        expected_mortar_data_history.emplace(mortar_id,
                                             boundary_history_type<2>{});
      }
      expected_mortar_data_history[mortar_id_a] = make_boundary_history(
          mortar_id_a, orig_mesh, expected_mortar_meshes.at(mortar_id_a),
          make_array<2>(Spectral::SegmentSize::Full),
          {{Spectral::SegmentSize::Full}}, false, true);
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_single_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  {
    INFO("Split - nw");
    auto box =
        db::create<tmpl::push_front<decltype(orig_single_items)::tags_list,
                                    domain::Tags::Domain<2>>>();

    const std::array mortar_ids{mortar_id_a, mortar_id_f, mortar_id_nw_ne,
                                mortar_id_nw_sw};
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = Element<2>(
              id_nw, {{mortar_id_a.direction(),
                       Neighbors<2>({mortar_id_a.id()}, rotated)},
                      {mortar_id_f.direction(),
                       Neighbors<2>({mortar_id_f.id()}, rotated)},
                      {mortar_id_nw_ne.direction(),
                       Neighbors<2>({mortar_id_nw_ne.id()}, aligned)},
                      {mortar_id_nw_sw.direction(),
                       Neighbors<2>({mortar_id_nw_sw.id()}, aligned)}});
          *mesh = orig_mesh;
          *neighbor_meshes = {
              {mortar_id_a, refined_single_neighbor_meshes.at(mortar_id_a)},
              {mortar_id_f, refined_single_neighbor_meshes.at(mortar_id_f)},
              {mortar_id_nw_ne, orig_mesh},
              {mortar_id_nw_sw, orig_mesh}};
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(make_not_null(&box),
                                              orig_single_items);

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_a, lgl_mesh<1>(4)},
        {mortar_id_f, lgl_mesh<1>(4)},
        {mortar_id_nw_ne, lgl_mesh<1>(4)},
        {mortar_id_nw_sw, lgl_mesh<1>(3)}};

    const ::dg::MortarMap<2, MortarInfo<2>> expected_mortar_infos{
        {mortar_id_a,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_f,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_nw_ne,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_nw_sw,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      for (const auto& mortar_id : mortar_ids) {
        expected_mortar_data_history.emplace(mortar_id,
                                             boundary_history_type<2>{});
      }
      expected_mortar_data_history[mortar_id_a] = make_boundary_history(
          mortar_id_a, orig_mesh, expected_mortar_meshes.at(mortar_id_a),
          element_size_nw, {{Spectral::SegmentSize::Full}}, false, true);
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  {
    INFO("Split - ne");
    auto box =
        db::create<tmpl::push_front<decltype(orig_single_items)::tags_list,
                                    domain::Tags::Domain<2>>>();

    const std::array mortar_ids{mortar_id_a, mortar_id_ne_nw, mortar_id_g,
                                mortar_id_ne_se};
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = Element<2>(
              id_ne, {{mortar_id_a.direction(),
                       Neighbors<2>({mortar_id_a.id()}, rotated)},
                      {mortar_id_ne_nw.direction(),
                       Neighbors<2>({mortar_id_ne_nw.id()}, aligned)},
                      {mortar_id_g.direction(),
                       Neighbors<2>({mortar_id_g.id()}, aligned)},
                      {mortar_id_ne_se.direction(),
                       Neighbors<2>({mortar_id_ne_se.id()}, aligned)}});
          *mesh = orig_mesh;
          *neighbor_meshes = {
              {mortar_id_a, refined_single_neighbor_meshes.at(mortar_id_a)},
              {mortar_id_ne_nw, orig_mesh},
              {mortar_id_g, refined_single_neighbor_meshes.at(mortar_id_g)},
              {mortar_id_ne_se, orig_mesh}};
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(make_not_null(&box),
                                              orig_single_items);

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_a, lgl_mesh<1>(4)},
        {mortar_id_ne_nw, lgl_mesh<1>(4)},
        {mortar_id_g, lgl_mesh<1>(4)},
        {mortar_id_ne_se, lgl_mesh<1>(3)}};

    const ::dg::MortarMap<2, MortarInfo<2>> expected_mortar_infos{
        {mortar_id_a,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_ne_nw,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_g,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_ne_se,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      for (const auto& mortar_id : mortar_ids) {
        expected_mortar_data_history.emplace(mortar_id,
                                             boundary_history_type<2>{});
      }
      expected_mortar_data_history[mortar_id_a] = make_boundary_history(
          mortar_id_a, orig_mesh, expected_mortar_meshes.at(mortar_id_a),
          element_size_ne, {{Spectral::SegmentSize::Full}}, false, true);
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  {
    INFO("Split - sw");
    auto box =
        db::create<tmpl::push_front<decltype(orig_single_items)::tags_list,
                                    domain::Tags::Domain<2>>>();

    const std::array mortar_ids{mortar_id_sw_nw, mortar_id_f, mortar_id_sw_se,
                                mortar_id_i};
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = Element<2>(
              id_sw, {{mortar_id_sw_nw.direction(),
                       Neighbors<2>({mortar_id_sw_nw.id()}, aligned)},
                      {mortar_id_f.direction(),
                       Neighbors<2>({mortar_id_f.id()}, rotated)},
                      {mortar_id_sw_se.direction(),
                       Neighbors<2>({mortar_id_sw_se.id()}, aligned)},
                      {mortar_id_i.direction(),
                       Neighbors<2>({mortar_id_i.id()}, aligned)}});
          *mesh = orig_mesh;
          *neighbor_meshes = {
              {mortar_id_sw_nw, orig_mesh},
              {mortar_id_f, refined_single_neighbor_meshes.at(mortar_id_f)},
              {mortar_id_sw_se, orig_mesh},
              {mortar_id_i, refined_single_neighbor_meshes.at(mortar_id_i)}};
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(make_not_null(&box),
                                              orig_single_items);

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_sw_nw, lgl_mesh<1>(3)},
        {mortar_id_f, lgl_mesh<1>(4)},
        {mortar_id_sw_se, lgl_mesh<1>(4)},
        {mortar_id_i, lgl_mesh<1>(5)}};

    const ::dg::MortarMap<2, MortarInfo<2>> expected_mortar_infos{
        {mortar_id_sw_nw,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_f,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::OrientCopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_sw_se,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_i,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      for (const auto& mortar_id : mortar_ids) {
        expected_mortar_data_history.emplace(mortar_id,
                                             boundary_history_type<2>{});
      }
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }

  {
    INFO("Split - se");
    auto box =
        db::create<tmpl::push_front<decltype(orig_single_items)::tags_list,
                                    domain::Tags::Domain<2>>>();

    const std::array mortar_ids{mortar_id_se_ne, mortar_id_se_sw, mortar_id_h,
                                mortar_id_i};
    db::mutate<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
               domain::Tags::NeighborMesh<2>>(
        [&](const gsl::not_null<Element<2>*> element,
            const gsl::not_null<Mesh<2>*> mesh,
            const gsl::not_null<::dg::MortarMap<2, Mesh<2>>*> neighbor_meshes) {
          *element = Element<2>(
              id_se, {{mortar_id_se_ne.direction(),
                       Neighbors<2>({mortar_id_se_ne.id()}, aligned)},
                      {mortar_id_se_sw.direction(),
                       Neighbors<2>({mortar_id_se_sw.id()}, aligned)},
                      {mortar_id_g.direction(),
                       Neighbors<2>({mortar_id_h.id()}, aligned)},
                      {mortar_id_i.direction(),
                       Neighbors<2>({mortar_id_i.id()}, aligned)}});
          *mesh = orig_mesh;
          *neighbor_meshes = {
              {mortar_id_se_ne, orig_mesh},
              {mortar_id_se_sw, orig_mesh},
              {mortar_id_h, refined_single_neighbor_meshes.at(mortar_id_h)},
              {mortar_id_i, refined_single_neighbor_meshes.at(mortar_id_i)}};
        },
        make_not_null(&box));

    db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
        Metavariables<2, LocalTimeStepping>>>(make_not_null(&box),
                                              orig_single_items);

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {mortar_id_se_ne, lgl_mesh<1>(3)},
        {mortar_id_se_sw, lgl_mesh<1>(4)},
        {mortar_id_h, lgl_mesh<1>(4)},
        {mortar_id_i, lgl_mesh<1>(5)}};

    const ::dg::MortarMap<2, MortarInfo<2>> expected_mortar_infos{
        {mortar_id_se_ne,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_se_sw,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_h,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}},
        {mortar_id_i,
         MortarInfo<2>{
             {.mortar_size = {{Spectral::SegmentSize::Full}},
              .interface_data_policy = InterfaceDataPolicy::CopyProject,
              .time_stepping_policy = time_stepping_policy}}}};

    mortar_data_history_tag::type expected_mortar_data_history{};
    if (LocalTimeStepping) {
      for (const auto& mortar_id : mortar_ids) {
        expected_mortar_data_history.emplace(mortar_id,
                                             boundary_history_type<2>{});
      }
    }

    CHECK(db::get<Tags::MortarData<2>>(box) == empty_mortar_data(mortar_ids));
    CHECK(db::get<Tags::MortarMesh<2>>(box) == expected_mortar_meshes);
    CHECK(db::get<Tags::MortarInfo<2>>(box) == expected_mortar_infos);
    CHECK(db::get<Tags::MortarNextTemporalId<2>>(box) ==
          constant_next_temporal_ids(mortar_ids, temporal_id));
    CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<2>>(box) ==
          empty_normal_covector_and_magnitude);
    check_boundary_histories(db::get<mortar_data_history_tag>(box),
                             expected_mortar_data_history);
  }
}

void test_h_refinement_mortar_sizes_local_impl(
    const std::vector<SegmentId>& pre_xi, const std::vector<SegmentId>& pre_eta,
    const std::vector<SegmentId>& post_xi,
    const std::vector<SegmentId>& post_eta,
    const OrientationMap<3>& orientation) {
  using NormalVars =
      Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                           evolution::dg::Tags::NormalCovector<3>>>;
  using mortar_data_history_tag =
      Tags::MortarDataHistory<3, dt_variables_tag<3>::type>;

  const ElementId<3> self_id(1, {{{1, 0}, {1, 0}, {0, 0}}});
  const auto direction = Direction<3>::upper_zeta();
  const Mesh<3> mesh = lgl_mesh<3>(4);
  const Mesh<2> mortar_mesh = lgl_mesh<2>(4);
  const TimeStepId time_step_id(true, 5, Slab(1.2, 3.4).start());

  // Pre-refinement data
  ::dg::MortarMap<3, evolution::dg::MortarDataHolder<3>> mortar_data{};
  ::dg::MortarMap<3, Mesh<2>> mortar_meshes{};
  ::dg::MortarMap<3, MortarInfo<3>> mortar_infos{};
  ::dg::MortarMap<3, TimeStepId> mortar_next_temporal_ids{};
  // NOLINTNEXTLINE(misc-const-correctness) - false positive - object is moved
  DirectionMap<3, std::optional<NormalVars>> normal_covector_and_magnitude{
      {direction, std::nullopt}};
  mortar_data_history_tag::type mortar_data_history{};
  std::unordered_set<ElementId<3>> old_neighbors{};
  for (const auto& segment_xi : pre_xi) {
    for (const auto& segment_eta : pre_eta) {
      const ElementId<3> neighbor(
          2, orientation(std::array{segment_xi, segment_eta, SegmentId{0, 0}}));
      const DirectionalId mortar_id{direction, neighbor};
      const auto mortar_size =
          ::dg::mortar_size(self_id, neighbor, 2, orientation);
      old_neighbors.emplace(neighbor);
      mortar_data.emplace(mortar_id, evolution::dg::MortarDataHolder<3>{});
      mortar_meshes.emplace(mortar_id, mortar_mesh);
      mortar_infos.emplace(
          mortar_id,
          MortarInfo<3>{
              {.mortar_size = mortar_size,
               .interface_data_policy =
                   orientation.is_aligned()
                       ? InterfaceDataPolicy::CopyProject
                       : InterfaceDataPolicy::OrientCopyProject,
               .time_stepping_policy = TimeSteppingPolicy::Conservative}});
      mortar_next_temporal_ids.emplace(mortar_id, time_step_id);
      mortar_data_history.emplace(
          mortar_id,
          make_boundary_history(mortar_id, mesh, mortar_mesh,
                                make_array<3>(Spectral::SegmentSize::Full),
                                mortar_size, true, true));
    }
  }
  const Element<3> old_element(
      self_id,
      {{direction, Neighbors<3>(std::move(old_neighbors), orientation)}});

  // Post-refinement data
  std::unordered_set<ElementId<3>> neighbors{};
  ::dg::MortarMap<3, Mesh<3>> neighbor_meshes{};
  mortar_data_history_tag::type expected_mortar_data_history{};
  for (const auto& segment_xi : post_xi) {
    for (const auto& segment_eta : post_eta) {
      const ElementId<3> neighbor(
          2, orientation(std::array{segment_xi, segment_eta, SegmentId{0, 0}}));
      const DirectionalId mortar_id{direction, neighbor};
      const auto mortar_size =
          ::dg::mortar_size(self_id, neighbor, 2, orientation);
      neighbors.emplace(neighbor);
      neighbor_meshes.emplace(mortar_id, mesh);
      expected_mortar_data_history.emplace(
          mortar_id,
          make_boundary_history(mortar_id, mesh, mortar_mesh,
                                make_array<3>(Spectral::SegmentSize::Full),
                                mortar_size, true,
                                pre_xi == post_xi and pre_eta == post_eta));
    }
  }
  // NOLINTNEXTLINE(misc-const-correctness) - false positive - object is moved
  Element<3> element(
      self_id, {{direction, Neighbors<3>(std::move(neighbors), orientation)}});

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Domain<3>, Tags::MortarData<3>, Tags::MortarMesh<3>,
      Tags::MortarInfo<3>, Tags::MortarNextTemporalId<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      mortar_data_history_tag, domain::Tags::Mesh<3>, domain::Tags::Element<3>,
      domain::Tags::NeighborMesh<3>, ::Tags::TimeStepId>>(
      Domain<3>{}, std::move(mortar_data), std::move(mortar_meshes),
      std::move(mortar_infos), std::move(mortar_next_temporal_ids),
      std::move(normal_covector_and_magnitude), std::move(mortar_data_history),
      mesh, std::move(element), std::move(neighbor_meshes), time_step_id);

  db::mutate_apply<
      evolution::dg::Initialization::ProjectMortars<Metavariables<3, true>>>(
      make_not_null(&box), std::pair(mesh, old_element));

  check_boundary_histories(db::get<mortar_data_history_tag>(box),
                           expected_mortar_data_history);
}

// Test projections of local mortar data in different 3D mortar configurations.
void test_h_refinement_mortar_sizes_local() {
  // In each tangential dimension, neighbors can change (or not) in
  // five ways, left-to-right for splitting, right-to-left for joining:
  const std::vector<std::pair<std::vector<SegmentId>, std::vector<SegmentId>>>
      dimension_cases{{{{0, 0}}, {{0, 0}}},
                      {{{0, 0}}, {{1, 0}}},
                      {{{1, 0}}, {{1, 0}}},
                      {{{1, 0}}, {{2, 0}, {2, 1}}},
                      {{{2, 0}, {2, 1}}, {{2, 0}, {2, 1}}}};
  const std::vector<OrientationMap<3>> orientations{
      OrientationMap<3>::create_aligned(),
      OrientationMap<3>(std::array{Direction<3>::lower_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::upper_zeta()}),
      OrientationMap<3>(std::array{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}),
      OrientationMap<3>(std::array{Direction<3>::upper_eta(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::upper_xi()})};

  for (const auto& [large_xi, small_xi] : dimension_cases) {
    for (const auto& [large_eta, small_eta] : dimension_cases) {
      for (const auto& orientation : orientations) {
        // Split
        test_h_refinement_mortar_sizes_local_impl(large_xi, large_eta, small_xi,
                                                  small_eta, orientation);
        // Join
        test_h_refinement_mortar_sizes_local_impl(small_xi, small_eta, large_xi,
                                                  large_eta, orientation);
      }
    }
  }
}

void test_h_refinement_mortar_sizes_remote_impl_split(
    const SegmentId& pre_xi, const SegmentId& pre_eta, const SegmentId& post_xi,
    const SegmentId& post_eta, const OrientationMap<3>& orientation) {
  using mortar_data_history_tag =
      Tags::MortarDataHistory<3, dt_variables_tag<3>::type>;

  const auto direction = Direction<3>::upper_zeta();
  const Mesh<3> mesh = lgl_mesh<3>(4);
  const Mesh<2> mortar_mesh = lgl_mesh<2>(4);
  const TimeStepId time_step_id(true, 5, Slab(1.2, 3.4).start());

  const std::array neighbor_segments{SegmentId{1, 0}, SegmentId{1, 1}};

  const ElementId<3> parent_id(1, {{pre_xi, pre_eta, {0, 0}}});
  const ElementId<3> self_id(1, {{post_xi, post_eta, {0, 0}}});
  CAPTURE(parent_id);
  CAPTURE(self_id);

  // Pre-refinement data
  mortar_data_history_tag::type parent_mortar_data_history{};
  std::unordered_set<ElementId<3>> parent_neighbors{};
  for (const auto& segment_xi : neighbor_segments) {
    if (not overlapping(segment_xi, pre_xi)) {
      continue;
    }
    for (const auto& segment_eta : neighbor_segments) {
      if (not overlapping(segment_eta, pre_eta)) {
        continue;
      }
      const ElementId<3> neighbor(
          2, orientation(std::array{segment_xi, segment_eta, SegmentId{0, 0}}));
      const DirectionalId mortar_id{direction, neighbor};
      const auto mortar_size =
          ::dg::mortar_size(parent_id, neighbor, 2, orientation);
      parent_neighbors.emplace(neighbor);
      parent_mortar_data_history.emplace(
          mortar_id,
          make_boundary_history(mortar_id, mesh, mortar_mesh,
                                make_array<3>(Spectral::SegmentSize::Full),
                                mortar_size, true, true));
    }
  }
  Element<3> parent_element(
      parent_id,
      {{direction, Neighbors<3>(std::move(parent_neighbors), orientation)}});

  const tuples::TaggedTuple<domain::Tags::Element<3>, ::Tags::TimeStepId,
                            mortar_data_history_tag>
      parent_items{std::move(parent_element), time_step_id,
                   std::move(parent_mortar_data_history)};

  // Post-refinement data
  std::unordered_set<ElementId<3>> neighbors{};
  ::dg::MortarMap<3, Mesh<3>> neighbor_meshes{};
  mortar_data_history_tag::type expected_mortar_data_history{};
  for (const auto& segment_xi : neighbor_segments) {
    if (not overlapping(segment_xi, post_xi)) {
      continue;
    }
    for (const auto& segment_eta : neighbor_segments) {
      if (not overlapping(segment_eta, post_eta)) {
        continue;
      }
      const ElementId<3> neighbor(
          2, orientation(std::array{segment_xi, segment_eta, SegmentId{0, 0}}));
      const DirectionalId mortar_id{direction, neighbor};
      const auto element_size =
          domain::child_size(self_id.segment_ids(), parent_id.segment_ids());
      const auto mortar_size =
          ::dg::mortar_size(self_id, neighbor, 2, orientation);
      neighbors.emplace(neighbor);
      neighbor_meshes.emplace(mortar_id, mesh);
      expected_mortar_data_history.emplace(
          mortar_id,
          make_boundary_history(mortar_id, mesh, mortar_mesh, element_size,
                                mortar_size, false, true));
    }
  }
  // NOLINTNEXTLINE(misc-const-correctness) - false positive - object is moved
  Element<3> element(
      self_id, {{direction, Neighbors<3>(std::move(neighbors), orientation)}});

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Domain<3>, Tags::MortarData<3>, Tags::MortarMesh<3>,
      Tags::MortarInfo<3>, Tags::MortarNextTemporalId<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      mortar_data_history_tag, domain::Tags::Mesh<3>, domain::Tags::Element<3>,
      domain::Tags::NeighborMesh<3>, ::Tags::TimeStepId>>(
      Domain<3>{}, Tags::MortarData<3>::type{}, Tags::MortarMesh<3>::type{},
      Tags::MortarInfo<3>::type{}, Tags::MortarNextTemporalId<3>::type{},
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type{},
      mortar_data_history_tag::type{}, mesh, std::move(element),
      std::move(neighbor_meshes), time_step_id);

  db::mutate_apply<
      evolution::dg::Initialization::ProjectMortars<Metavariables<3, true>>>(
      make_not_null(&box), parent_items);

  check_boundary_histories(db::get<mortar_data_history_tag>(box),
                           expected_mortar_data_history);
}

void test_h_refinement_mortar_sizes_remote_impl_join(
    const std::vector<SegmentId>& pre_xi, const std::vector<SegmentId>& pre_eta,
    const SegmentId& post_xi, const SegmentId& post_eta,
    const OrientationMap<3>& orientation) {
  using mortar_data_history_tag =
      Tags::MortarDataHistory<3, dt_variables_tag<3>::type>;

  const auto direction = Direction<3>::upper_zeta();
  const Mesh<3> mesh = lgl_mesh<3>(4);
  const Mesh<2> mortar_mesh = lgl_mesh<2>(4);
  const TimeStepId time_step_id(true, 5, Slab(1.2, 3.4).start());

  const std::array neighbor_segments{SegmentId{1, 0}, SegmentId{1, 1}};

  const ElementId<3> self_id(1, {{post_xi, post_eta, {0, 0}}});

  // Pre-refinement data
  using ChildItems =
      tuples::TaggedTuple<::Tags::TimeStepId, mortar_data_history_tag>;
  std::unordered_map<ElementId<3>, ChildItems> children_items{};
  for (const auto& segment_xi : pre_xi) {
    for (const auto& segment_eta : pre_eta) {
      const ElementId<3> child_id(1, {{segment_xi, segment_eta, {0, 0}}});
      const auto child_size =
          domain::child_size(child_id.segment_ids(), self_id.segment_ids());
      mortar_data_history_tag::type mortar_data_history{};
      for (const auto& neighbor_xi : neighbor_segments) {
        if (not overlapping(neighbor_xi, segment_xi)) {
          continue;
        }
        for (const auto& neighbor_eta : neighbor_segments) {
          if (not overlapping(neighbor_eta, segment_eta)) {
            continue;
          }
          const ElementId<3> neighbor(
              2, orientation(
                     std::array{neighbor_xi, neighbor_eta, SegmentId{0, 0}}));
          const DirectionalId mortar_id{direction, neighbor};
          const auto mortar_size =
              ::dg::mortar_size(child_id, neighbor, 2, orientation);
          mortar_data_history.emplace(
              mortar_id,
              make_boundary_history(mortar_id, mesh, mortar_mesh, child_size,
                                    mortar_size, true, true));
        }
      }
      children_items.emplace(
          child_id, ChildItems{time_step_id, std::move(mortar_data_history)});
    }
  }

  // Post-refinement data
  std::unordered_set<ElementId<3>> neighbors{};
  ::dg::MortarMap<3, Mesh<3>> neighbor_meshes{};
  mortar_data_history_tag::type expected_mortar_data_history{};
  for (const auto& segment_xi : neighbor_segments) {
    if (not overlapping(segment_xi, post_xi)) {
      continue;
    }
    for (const auto& segment_eta : neighbor_segments) {
      if (not overlapping(segment_eta, post_eta)) {
        continue;
      }
      const ElementId<3> neighbor(
          2, orientation(std::array{segment_xi, segment_eta, SegmentId{0, 0}}));
      const DirectionalId mortar_id{direction, neighbor};
      const auto mortar_size =
          ::dg::mortar_size(self_id, neighbor, 2, orientation);
      neighbors.emplace(neighbor);
      neighbor_meshes.emplace(mortar_id, mesh);
      expected_mortar_data_history.emplace(
          mortar_id,
          make_boundary_history(mortar_id, mesh, mortar_mesh,
                                make_array<3>(Spectral::SegmentSize::Full),
                                mortar_size, false, true));
    }
  }
  // NOLINTNEXTLINE(misc-const-correctness) - false positive - object is moved
  Element<3> element(
      self_id, {{direction, Neighbors<3>(std::move(neighbors), orientation)}});

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Domain<3>, Tags::MortarData<3>, Tags::MortarMesh<3>,
      Tags::MortarInfo<3>, Tags::MortarNextTemporalId<3>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>,
      mortar_data_history_tag, domain::Tags::Mesh<3>, domain::Tags::Element<3>,
      domain::Tags::NeighborMesh<3>, ::Tags::TimeStepId>>(
      Domain<3>{}, Tags::MortarData<3>::type{}, Tags::MortarMesh<3>::type{},
      Tags::MortarInfo<3>::type{}, Tags::MortarNextTemporalId<3>::type{},
      evolution::dg::Tags::NormalCovectorAndMagnitude<3>::type{},
      mortar_data_history_tag::type{}, mesh, std::move(element),
      std::move(neighbor_meshes), time_step_id);

  db::mutate_apply<
      evolution::dg::Initialization::ProjectMortars<Metavariables<3, true>>>(
      make_not_null(&box), children_items);

  check_boundary_histories(db::get<mortar_data_history_tag>(box),
                           expected_mortar_data_history);
}

// Test projections of remote data in different 3D mortar configurations.
void test_h_refinement_mortar_sizes_remote() {
  // In each tangential dimension, our element can change (or not) in
  // five ways, left-to-right for splitting, right-to-left for
  // joining.  In each case, we will just check the mortar with the
  // element with xi-eta segments {{1, 0}, {1, 0}}.
  const std::vector<std::pair<SegmentId, std::vector<SegmentId>>>
      dimension_cases{{{0, 0}, {{0, 0}}},
                      {{0, 0}, {{1, 0}, {1, 1}}},
                      {{1, 0}, {{1, 0}}},
                      {{1, 0}, {{2, 0}, {2, 1}}},
                      {{2, 0}, {{2, 0}}}};
  const std::vector<OrientationMap<3>> orientations{
      OrientationMap<3>::create_aligned(),
      OrientationMap<3>(std::array{Direction<3>::lower_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::upper_zeta()}),
      OrientationMap<3>(std::array{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}),
      OrientationMap<3>(std::array{Direction<3>::upper_eta(),
                                   Direction<3>::upper_zeta(),
                                   Direction<3>::upper_xi()})};

  for (const auto& [large_xi, small_xis] : dimension_cases) {
    for (const auto& [large_eta, small_etas] : dimension_cases) {
      if (small_xis.size() == 1 and small_etas.size() == 1) {
        continue;
      }
      for (const auto& orientation : orientations) {
        test_h_refinement_mortar_sizes_remote_impl_split(
            large_xi, large_eta, small_xis.front(), small_etas.front(),
            orientation);
        test_h_refinement_mortar_sizes_remote_impl_join(
            small_xis, small_etas, large_xi, large_eta, orientation);
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Initialization.Mortars",
                  "[Unit][Evolution]") {
  for (const auto quadrature :
       {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
    Test<1, true>::apply(quadrature);
    Test<2, true>::apply(quadrature);
    Test<3, true>::apply(quadrature);

    Test<1, false>::apply(quadrature);
    Test<2, false>::apply(quadrature);
    Test<3, false>::apply(quadrature);
  }

  domain::creators::register_derived_with_charm();
  test_nonconforming_blocks<false>();

  static_assert(
      tt::assert_conforms_to_v<evolution::dg::Initialization::ProjectMortars<
                                   Metavariables<1, false>>,
                               amr::protocols::Projector>);
  test_p_refine_gts<1>();
  test_p_refine_gts<2>();
  test_p_refine_gts<3>();
  test_p_refine_lts<1>();
  test_p_refine_lts<2>();
  test_p_refine_lts<3>();
  test_h_refinement<false>();
  test_h_refinement<true>();
  test_h_refinement_mortar_sizes_local();
  test_h_refinement_mortar_sizes_remote();
}
}  // namespace evolution::dg
