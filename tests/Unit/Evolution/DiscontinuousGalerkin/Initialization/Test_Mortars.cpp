// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
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
                 evolution::dg::Tags::Quadrature>;
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
  using const_global_cache_tags = tmpl::list<domain::Tags::InitialExtents<Dim>>;
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
    const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_meshes,
    const ::dg::MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>&
        expected_mortar_sizes,
    const DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>&
        expected_normal_covector_quantities) {
  using metavars = Metavariables<Dim, LocalTimeStepping>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{initial_extents};
  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, element.id(),
      {time_step_id, next_time_step_id, element,
       domain::Initialization::create_initial_mesh(
           initial_extents, element.id(), quadrature, {}),
       quadrature});

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
  const auto& mortar_sizes = get_tag(Tags::MortarSize<Dim>{});
  CHECK(mortar_sizes == expected_mortar_sizes);
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

  // Cast result of `operator==` to a bool to trick Catch into not trying to
  // stream a nested STL container.
  CHECK(static_cast<bool>(
      get_tag(evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>{}) ==
      expected_normal_covector_quantities));
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

    DirectionMap<1, Neighbors<1>> neighbors{};
    neighbors[Direction<1>::upper_xi()] = Neighbors<1>{{east_id}, {}};
    const Element<1> element{element_id, neighbors};
    const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    // We are working with 2 mortars here: a domain boundary at lower xi
    // and an interface at upper xi.
    const DirectionalId<1> interface_mortar_id{Direction<1>::upper_xi(),
                                               east_id};
    const ::dg::MortarMap<1, Mesh<0>> expected_mortar_meshes{
        {interface_mortar_id, {}}};
    const ::dg::MortarMap<1, std::array<Spectral::MortarSize, 0>>
        expected_mortar_sizes{{interface_mortar_id, {}}};

    const DirectionMap<
        1, std::optional<
               Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                    evolution::dg::Tags::NormalCovector<1>>>>>
        expected_normal_covector_quantities{{Direction<1>::lower_xi(), {}},
                                            {Direction<1>::upper_xi(), {}}};

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature,
                                 expected_mortar_meshes, expected_mortar_sizes,
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

    DirectionMap<2, Neighbors<2>> neighbors{};
    neighbors[Direction<2>::upper_xi()] = Neighbors<2>{{east_id}, {}};
    neighbors[Direction<2>::lower_eta()] = Neighbors<2>{{south_id}, {}};

    const Element<2> element{element_id, neighbors};
    const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    // We are working with 4 mortars here: the domain boundary west and north,
    // and interfaces south and east.
    const DirectionalId<2> interface_mortar_id_east{Direction<2>::upper_xi(),
                                                    east_id};
    const DirectionalId<2> interface_mortar_id_south{Direction<2>::lower_eta(),
                                                     south_id};

    const ::dg::MortarMap<2, Mesh<1>> expected_mortar_meshes{
        {interface_mortar_id_east,
         Mesh<1>(2, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_south,
         Mesh<1>(3, Spectral::Basis::Legendre, quadrature)}};
    ::dg::MortarMap<2, std::array<Spectral::MortarSize, 1>>
        expected_mortar_sizes{};
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

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature,
                                 expected_mortar_meshes, expected_mortar_sizes,
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
    const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {6, 100}}};

    const DirectionalId<3> interface_mortar_id_right{Direction<3>::upper_xi(),
                                                     right_id};
    const DirectionalId<3> interface_mortar_id_front{Direction<3>::lower_eta(),
                                                     front_id};
    const DirectionalId<3> interface_mortar_id_top{Direction<3>::upper_zeta(),
                                                   top_id};

    const ::dg::MortarMap<3, Mesh<2>> expected_mortar_meshes{
        {interface_mortar_id_right,
         Mesh<2>({{3, 4}}, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_front,
         Mesh<2>({{2, 4}}, Spectral::Basis::Legendre, quadrature)},
        {interface_mortar_id_top,
         Mesh<2>({{2, 3}}, Spectral::Basis::Legendre, quadrature)}};
    ::dg::MortarMap<3, std::array<Spectral::MortarSize, 2>>
        expected_mortar_sizes{};
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

    test_impl<LocalTimeStepping>(initial_extents, element, time_step_id,
                                 next_time_step_id, quadrature,
                                 expected_mortar_meshes, expected_mortar_sizes,
                                 expected_normal_covector_quantities);
  }
};

template <size_t Dim, bool UsingLts>
void test_p_refine(
    ::dg::MortarMap<Dim, evolution::dg::MortarData<Dim>>& mortar_data,
    ::dg::MortarMap<Dim, Mesh<Dim - 1>>& mortar_mesh,
    ::dg::MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>&
        mortar_size,
    ::dg::MortarMap<Dim, TimeStepId>& mortar_next_temporal_id,
    DirectionMap<Dim, std::optional<Variables<tmpl::list<
                          evolution::dg::Tags::MagnitudeOfNormal,
                          evolution::dg::Tags::NormalCovector<Dim>>>>>&
        normal_covector_and_magnitude,
    mortar_data_history_type<Dim>& mortar_data_history,
    const Mesh<Dim>& old_mesh, Mesh<Dim>& new_mesh,
    const Element<Dim>& old_element, Element<Dim>& new_element,
    std::unordered_map<ElementId<Dim>, amr::Info<Dim>>& neighbor_info,
    const ::dg::MortarMap<
        Dim, evolution::dg::MortarData<Dim>>& /*expected_mortar_data*/,
    const ::dg::MortarMap<Dim, Mesh<Dim - 1>>& expected_mortar_mesh,
    const ::dg::MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>&
        expected_mortar_size,
    const ::dg::MortarMap<Dim, TimeStepId>& expected_mortar_next_temporal_id,
    const DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                evolution::dg::Tags::MagnitudeOfNormal,
                                evolution::dg::Tags::NormalCovector<Dim>>>>>&
        expected_normal_covector_and_magnitude,
    const mortar_data_history_type<Dim>& expected_mortar_data_history) {
  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
      amr::Tags::NeighborInfo<Dim>, Tags::MortarData<Dim>,
      Tags::MortarMesh<Dim>, Tags::MortarSize<Dim>,
      Tags::MortarNextTemporalId<Dim>,
      evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>,
      Tags::MortarDataHistory<Dim, typename dt_variables_tag<Dim>::type>>>(
      std::move(new_mesh), std::move(new_element), std::move(neighbor_info),
      std::move(mortar_data), std::move(mortar_mesh), std::move(mortar_size),
      std::move(mortar_next_temporal_id),
      std::move(normal_covector_and_magnitude), std::move(mortar_data_history));

  db::mutate_apply<evolution::dg::Initialization::ProjectMortars<
      Metavariables<Dim, UsingLts>>>(make_not_null(&box),
                                     std::make_pair(old_mesh, old_element));

  // Can't check the state as a default constructed MortarData has a default
  // constructed TimeStepId which has nans which always compare as unequal
  // CHECK(db::get<Tags::MortarData<Dim>>(box) == expected_mortar_data);
  CHECK(db::get<Tags::MortarMesh<Dim>>(box) == expected_mortar_mesh);
  CHECK(db::get<Tags::MortarSize<Dim>>(box) == expected_mortar_size);
  CHECK(db::get<Tags::MortarNextTemporalId<Dim>>(box) ==
        expected_mortar_next_temporal_id);
  CHECK(db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(box) ==
        expected_normal_covector_and_magnitude);
  if (not UsingLts) {
    (void)(expected_mortar_data_history);
    CHECK(
        db::get<
            Tags::MortarDataHistory<Dim, typename dt_variables_tag<Dim>::type>>(
            box)
            .empty());
  }
}

template <size_t Dim>
Element<Dim> make_element();

template <>
Element<1> make_element<1>() {
  const ElementId<1> element_id{0, {{SegmentId{2, 0}}}};
  const ElementId<1> neighbor_id{0, {{SegmentId{2, 1}}}};
  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors[Direction<1>::upper_xi()] = Neighbors<1>{{neighbor_id}, {}};
  return Element<1>{element_id, neighbors};
}

template <>
Element<2> make_element<2>() {
  const ElementId<2> element_id{0, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
  const ElementId<2> east_id(0, {{SegmentId{1, 1}, SegmentId{1, 1}}});
  const ElementId<2> south_id(0, {{SegmentId{1, 0}, SegmentId{1, 0}}});
  DirectionMap<2, Neighbors<2>> neighbors{};
  neighbors[Direction<2>::upper_xi()] = Neighbors<2>{{east_id}, {}};
  neighbors[Direction<2>::lower_eta()] = Neighbors<2>{{south_id}, {}};
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
  neighbors[Direction<3>::upper_xi()] = Neighbors<3>{{right_id}, {}};
  neighbors[Direction<3>::lower_eta()] = Neighbors<3>{{front_id}, {}};
  neighbors[Direction<3>::upper_zeta()] = Neighbors<3>{{top_id}, {}};
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

  // These quantities are re-allocated after projection, so we can
  // just set them to empty maps...
  ::dg::MortarMap<Dim, evolution::dg::MortarData<Dim>> mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> mortar_mesh{};
  ::dg::MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>> mortar_size{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_and_magnitude{};
  mortar_data_history_type<Dim> mortar_data_history{};

  ::dg::MortarMap<Dim, TimeStepId> mortar_next_temporal_ids{};
  std::unordered_map<ElementId<Dim>, amr::Info<Dim>> neighbor_info{};
  for (const auto& [direction, neighbors] : old_element.neighbors()) {
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
      neighbor_info.emplace(
          neighbor,
          amr::Info<Dim>{std::array<amr::Flag, Dim>{}, neighbor_mesh});
    }
  }

  ::dg::MortarMap<Dim, evolution::dg::MortarData<Dim>> expected_mortar_data{};
  ::dg::MortarMap<Dim, Mesh<Dim - 1>> expected_mortar_mesh{};
  ::dg::MortarMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>
      expected_mortar_size{};
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
      expected_mortar_data.emplace(mortar_id, MortarData<Dim>{1});
      expected_mortar_mesh.emplace(
          mortar_id,
          ::dg::mortar_mesh(new_mesh.slice_away(direction.dimension()),
                            neighbor_info.at(neighbor).new_mesh.slice_away(
                                direction.dimension())));
      expected_mortar_size.emplace(
          mortar_id,
          ::dg::mortar_size(new_element.id(), neighbor, direction.dimension(),
                            neighbors.orientation()));
      expected_mortar_next_temporal_ids.emplace(mortar_id, next_temporal_id);
    }
  }
  for (const auto& direction : new_element.external_boundaries()) {
    expected_normal_covector_and_magnitude[direction] = std::nullopt;
  }

  test_p_refine<Dim, false>(
      mortar_data, mortar_mesh, mortar_size, mortar_next_temporal_ids,
      normal_covector_and_magnitude, mortar_data_history, old_mesh, new_mesh,
      old_element, new_element, neighbor_info, expected_mortar_data,
      expected_mortar_mesh, expected_mortar_size,
      expected_mortar_next_temporal_ids, expected_normal_covector_and_magnitude,
      expected_mortar_data_history);
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
  static_assert(
      tt::assert_conforms_to_v<evolution::dg::Initialization::ProjectMortars<
                                   Metavariables<1, false>>,
                               amr::protocols::Projector>);
  test_p_refine_gts<1>();
  test_p_refine_gts<2>();
  test_p_refine_gts<3>();
}
}  // namespace evolution::dg
