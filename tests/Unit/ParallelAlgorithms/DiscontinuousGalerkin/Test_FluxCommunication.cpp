// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

using TemporalId = int;

struct TemporalIdTag : db::SimpleTag {
  using type = TemporalId;
};

template <size_t Dim>
struct TestBoundaryData {
  ElementId<Dim> element_id{};
  bool is_projected = false;
  bool is_oriented = false;
  TestBoundaryData project_to_mortar(
      const Mesh<Dim - 1>& /*face_mesh*/, const Mesh<Dim - 1>& /*mortar_mesh*/,
      const std::array<Spectral::MortarSize, Dim - 1>& /*mortar_size*/) const
      noexcept {
    return {element_id, true, is_oriented};
  }
  void orient_on_slice(
      const Index<Dim - 1>& /*slice_extents*/, const size_t /*sliced_dim*/,
      const OrientationMap<Dim>& /*orientation_of_neighbor*/) noexcept {
    this->is_oriented = true;
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept {
    p | element_id;
    p | is_projected;
    p | is_oriented;
  }
};

template <size_t Dim>
using MortarData = dg::SimpleMortarData<TemporalId, TestBoundaryData<Dim>,
                                        TestBoundaryData<Dim>>;

template <size_t Dim>
struct MortarDataTag : db::SimpleTag {
  using type = MortarData<Dim>;
};

template <size_t Dim>
struct DgBoundaryScheme {
  static constexpr size_t volume_dim = Dim;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = temporal_id_tag;
  using mortar_data_tag = MortarDataTag<Dim>;
  using BoundaryData = TestBoundaryData<Dim>;
  struct boundary_data_computer {
    using argument_tags = tmpl::list<domain::Tags::Element<Dim>>;
    using volume_tags = tmpl::list<domain::Tags::Element<Dim>>;
    static BoundaryData apply(const Element<Dim>& element) noexcept {
      return {element.id(), false, false};
    }
  };
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<>;

  using simple_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, TemporalIdTag,
                 domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::ElementMap<Dim>>;
  using compute_tags = tmpl::list<
      domain::Tags::InternalDirections<Dim>,
      domain::Tags::BoundaryDirectionsInterior<Dim>,
      domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<Dim>,
                                     domain::Tags::Direction<Dim>>,
      domain::Tags::InterfaceCompute<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          domain::Tags::Direction<Dim>>,
      domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<Dim>,
                                     domain::Tags::InterfaceMesh<Dim>>,
      domain::Tags::InterfaceCompute<
          domain::Tags::BoundaryDirectionsInterior<Dim>,
          domain::Tags::InterfaceMesh<Dim>>>;
  using init_mortars_tags =
      tmpl::list<::Tags::Mortars<MortarDataTag<Dim>, Dim>,
                 ::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>>;
  using db_tags_list =
      tmpl::append<simple_tags, init_mortars_tags, compute_tags>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              // `InitializeMortars` needs to be in this phase and also needs
              // `RemoveOptionsAndTerminatePhase` because otherwise the action
              // testing framework can't compile `is_ready` for
              // `ReceiveDataForFluxes`. See
              // https://github.com/sxs-collaboration/spectre/issues/1908
              dg::Actions::InitializeMortars<DgBoundaryScheme<Dim>, false>,
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme<Dim>, domain::Tags::InternalDirections<Dim>>,
              dg::Actions::SendDataForFluxes<DgBoundaryScheme<Dim>>,
              dg::Actions::ReceiveDataForFluxes<DgBoundaryScheme<Dim>>,
              Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

// The mortars match the element (even though in one case the neighbor
// is larger).
void test_no_refinement() noexcept {
  using metavariables = Metavariables<2>;
  using element_array = ElementArray<2, metavariables>;
  using fluxes_inbox_tag = dg::FluxesInboxTag<DgBoundaryScheme<2>>;
  using all_mortar_data_tag = ::Tags::Mortars<MortarDataTag<2>, 2>;
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const std::vector<std::array<size_t, 2>> initial_extents{
      {{{3, 3}}, {{3, 3}}}};

  //      xi      Block       +- xi
  //      |     0   |   1     |
  // eta -+ +-------+-+-+---+ eta
  //        |       |X| |   |
  //        |       +-+-+   |
  //        |       | | |   |
  //        +-------+-+-+---+
  // We run the actions on the indicated element.  The blocks are square.
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> east_id(1, {{{2, 1}, {1, 0}}});
  const ElementId<2> south_id(1, {{{2, 0}, {1, 1}}});
  const dg::MortarId<2> mortar_id_west{Direction<2>::lower_xi(), west_id};
  const dg::MortarId<2> mortar_id_east{Direction<2>::upper_xi(), east_id};
  const dg::MortarId<2> mortar_id_south{Direction<2>::upper_eta(), south_id};
  const std::array<dg::MortarId<2>, 3> neighbor_mortar_ids{
      {mortar_id_west, mortar_id_east, mortar_id_south}};

  // OrientationMap from block 1 to block 0
  const OrientationMap<2> block_orientation(
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
      {{Direction<2>::lower_eta(), Direction<2>::lower_xi()}});

  // Since we're lazy and use the same map for both blocks (the
  // actions are only sensitive to the ElementMap, which does differ),
  // we need to make the xi and eta maps line up along the block
  // interface.
  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));

  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(xi_map, eta_map));

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  // Emplace self element
  {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());

    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, self_id,
        {initial_extents, time, mesh, element, std::move(map)});
  }

  const auto emplace_neighbor =
      [&initial_extents, &time, &mesh, &self_id, &coordmap, &runner](
          const ElementId<2>& id, const Direction<2>& direction,
          const OrientationMap<2>& orientation) noexcept {
        const Element<2> element(id, {{direction, {{self_id}, orientation}}});
        auto map = ElementMap<2, Frame::Inertial>(id, coordmap->get_clone());

        ActionTesting::emplace_component_and_initialize<element_array>(
            &runner, id,
            {initial_extents, time, mesh, element, std::move(map)});
      };

  emplace_neighbor(south_id, Direction<2>::lower_eta(), {});
  emplace_neighbor(east_id, Direction<2>::lower_xi(), {});
  emplace_neighbor(west_id, Direction<2>::lower_eta(),
                   block_orientation.inverse_map());
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, self_id);
  };

  // InitializeMortars on self
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // CollectDataForFluxes on self
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // SendDataForFluxes on self
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);

  // Here, we just check that messages are sent to the correct places.
  // We will check the received values on the central element later.
  {
    CHECK(runner.template nonempty_inboxes<element_array, fluxes_inbox_tag>() ==
          std::unordered_set<ElementId<2>>{west_id, east_id, south_id});
    const auto check_sent_data = [&runner, &self_id, &time](
                                     const ElementId<2>& id,
                                     const Direction<2>& direction) noexcept {
      const auto& flux_inbox =
          ActionTesting::get_inbox_tag<element_array, fluxes_inbox_tag>(runner,
                                                                        id);
      CHECK(flux_inbox.size() == 1);
      CHECK(flux_inbox.count(time) == 1);
      const auto& flux_inbox_at_time = flux_inbox.at(time);
      CHECK(flux_inbox_at_time.size() == 1);
      CHECK(flux_inbox_at_time.count({direction, self_id}) == 1);
    };
    check_sent_data(west_id, Direction<2>::lower_eta());
    check_sent_data(east_id, Direction<2>::lower_xi());
    check_sent_data(south_id, Direction<2>::lower_eta());
  }

  // Now check ReceiveDataForFluxes
  for (auto mortar_id : neighbor_mortar_ids) {
    CHECK_FALSE(ActionTesting::is_ready<element_array>(runner, self_id));
    // InitializeMortars on neighbor
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              mortar_id.second);
    // CollectDataForFluxes on neighbor
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              mortar_id.second);
    // SendDataForFluxes on neighbor
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              mortar_id.second);
  }
  CHECK(ActionTesting::is_ready<element_array>(runner, self_id));

  // ReceiveDataForFluxes on self
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  CHECK(ActionTesting::get_inbox_tag<element_array, fluxes_inbox_tag>(runner,
                                                                      self_id)
            .empty());

  auto mortar_history =
      serialize_and_deserialize(get_tag(all_mortar_data_tag{}));
  CHECK(mortar_history.size() == 3);
  const auto check_mortar = [&mortar_history, &self_id](
                                const dg::MortarId<2>& mortar_id,
                                const bool expect_projection,
                                const bool expect_orientation) noexcept {
    CAPTURE(mortar_id);
    const auto local_and_remote_data = mortar_history.at(mortar_id).extract();
    CHECK(local_and_remote_data.first.element_id == self_id);
    CHECK_FALSE(local_and_remote_data.first.is_projected);
    CHECK_FALSE(local_and_remote_data.first.is_oriented);
    CHECK(local_and_remote_data.second.element_id == mortar_id.second);
    CHECK(local_and_remote_data.second.is_projected == expect_projection);
    CHECK(local_and_remote_data.second.is_oriented == expect_orientation);
  };

  check_mortar(mortar_id_west, true, true);
  check_mortar(mortar_id_east, false, false);
  check_mortar(mortar_id_south, false, false);
}

void test_no_neighbors() noexcept {
  using metavariables = Metavariables<2>;
  using element_array = ElementArray<2, metavariables>;
  using all_mortar_data_tag = ::Tags::Mortars<MortarDataTag<2>, 2>;
  using fluxes_inbox_tag = dg::FluxesInboxTag<DgBoundaryScheme<2>>;

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const std::vector<std::array<size_t, 2>> initial_extents{
      {{{3, 3}}, {{3, 3}}}};
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});
  const Element<2> element(self_id, {});

  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));
  auto map = ElementMap<2, Frame::Inertial>(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D({-1., 1., 3., 7.}, {-1., 1., -2., 4.})));

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, self_id, {initial_extents, time, mesh, element, std::move(map)});
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // InitializeMortars
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // CollectDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // SendDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);

  CHECK(ActionTesting::get_databox_tag<element_array, all_mortar_data_tag>(
            runner, self_id)
            .empty());
  CHECK(runner.template nonempty_inboxes<element_array, fluxes_inbox_tag>()
            .empty());

  CHECK(ActionTesting::is_ready<element_array>(runner, self_id));

  // ReceiveDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);

  CHECK(ActionTesting::get_databox_tag<element_array, all_mortar_data_tag>(
            runner, self_id)
            .empty());
  CHECK(runner.template nonempty_inboxes<element_array, fluxes_inbox_tag>()
            .empty());
}

void test_p_refinement() noexcept {
  using metavariables = Metavariables<3>;
  using element_array = ElementArray<3, metavariables>;
  using all_mortar_data_tag = ::Tags::Mortars<MortarDataTag<3>, 3>;
  using fluxes_inbox_tag = dg::FluxesInboxTag<DgBoundaryScheme<3>>;

  const ElementId<3> self_id(0);
  const ElementId<3> neighbor_id(1);

  const Mesh<3> mesh_self({{2, 3, 4}}, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto);
  const Mesh<3> mesh_neighbor({{3, 3, 3}}, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
  const std::vector<std::array<size_t, 3>> initial_extents{
      {{{2, 3, 4}}, {{3, 3, 3}}}};

  const auto mortar_id = std::make_pair(Direction<3>::upper_eta(), neighbor_id);
  const Element<3> element(
      self_id, {{mortar_id.first,
                 {{neighbor_id},
                  OrientationMap<3>{
                      {{Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::lower_eta()}}}}}});

  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<3>>));
  ElementMap<3, Frame::Inertial> map(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{}));

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, self_id,
      {initial_extents, time, mesh_self, element, std::move(map)});
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, neighbor_id,
      {initial_extents, time, mesh_neighbor,
       // The following arguments are unused on the neighbor element, so their
       // values are irrelevant
       element, ElementMap<3, Frame::Inertial>{}});
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, self_id);
  };

  // InitializeMortars
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // CollectDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // SendDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);

  // Check local data
  {
    const auto& all_mortar_data = get_tag(all_mortar_data_tag{});
    CHECK(all_mortar_data.size() == 1);
    const auto& local_data = all_mortar_data.at(mortar_id).local_data(time);
    CHECK(local_data.element_id == self_id);
    CHECK(local_data.is_projected);
    CHECK_FALSE(local_data.is_oriented);
  }

  // Check sent data
  {
    CHECK(runner.template nonempty_inboxes<element_array, fluxes_inbox_tag>()
              .size() == 1);
    const auto& inbox =
        ActionTesting::get_inbox_tag<element_array, fluxes_inbox_tag>(
            runner, neighbor_id);
    const auto& received_data =
        inbox.at(time).at({Direction<3>::upper_xi(), self_id}).second;
    CHECK(received_data.element_id == self_id);
    CHECK(received_data.is_projected);
    CHECK(received_data.is_oriented);
  }
}

void test_h_refinement(const Spectral::MortarSize& mortar_size) {
  CAPTURE(mortar_size);
  using metavariables = Metavariables<2>;
  using element_array = ElementArray<2, metavariables>;
  using all_mortar_data_tag = ::Tags::Mortars<MortarDataTag<2>, 2>;
  using fluxes_inbox_tag = dg::FluxesInboxTag<DgBoundaryScheme<2>>;

  const ElementId<2> self_id(0);
  const auto neighbor_id = [&mortar_size]() -> ElementId<2> {
    switch (mortar_size) {
      case Spectral::MortarSize::Full:
        return {1, {{{0, 0}, {0, 0}}}};
      case Spectral::MortarSize::UpperHalf:
        return {1, {{{0, 0}, {1, 0}}}};
      case Spectral::MortarSize::LowerHalf:
        return {1, {{{0, 0}, {1, 1}}}};
      default:
        ERROR("Missing enum case");
    }
  }();

  const dg::MortarId<2> mortar_id{Direction<2>::upper_xi(), neighbor_id};
  const Element<2> element(
      self_id, {{mortar_id.first,
                 {{neighbor_id},
                  OrientationMap<2>{{{Direction<2>::upper_xi(),
                                      Direction<2>::lower_eta()}}}}}});

  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));
  ElementMap<2, Frame::Inertial> map(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D({-1., 1., -1., 1.}, {-1., 1., -1., 1.})));

  const Mesh<2> mesh(2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const std::vector<std::array<size_t, 2>> initial_extents{
      {{{2, 2}}, {{2, 2}}}};

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, self_id, {initial_extents, time, mesh, element, std::move(map)});
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, neighbor_id,
      {initial_extents, time, mesh,
       // The following arguments are unused on the neighbor element, so their
       // values are irrelevant
       element, ElementMap<2, Frame::Inertial>{}});
  runner.set_phase(metavariables::Phase::Testing);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, self_id);
  };

  // InitializeMortars
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // CollectDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  // SendDataForFluxes
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);

  // Check local data
  {
    const auto& all_mortar_sizes =
        get_tag(::Tags::Mortars<::Tags::MortarSize<1>, 2>{});
    CHECK(all_mortar_sizes.at(mortar_id) == dg::MortarSize<1>{{mortar_size}});
    const auto& all_mortar_data = get_tag(all_mortar_data_tag{});
    CHECK(all_mortar_data.size() == 1);
    const auto& local_data = all_mortar_data.at(mortar_id).local_data(time);
    CHECK(local_data.element_id == self_id);
    CHECK(local_data.is_projected ==
          (mortar_size != Spectral::MortarSize::Full));
    CHECK_FALSE(local_data.is_oriented);
  }

  // Check sent data
  {
    CHECK(runner.template nonempty_inboxes<element_array, fluxes_inbox_tag>()
              .size() == 1);
    const auto& inbox =
        ActionTesting::get_inbox_tag<element_array, fluxes_inbox_tag>(
            runner, neighbor_id);

    const auto& received_data =
        inbox.at(time).at({Direction<2>::lower_xi(), self_id}).second;
    CHECK(received_data.element_id == self_id);
    CHECK(received_data.is_projected ==
          (mortar_size != Spectral::MortarSize::Full));
    CHECK(received_data.is_oriented);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  test_no_refinement();
  test_no_neighbors();
  test_p_refinement();
  test_h_refinement(Spectral::MortarSize::Full);
  test_h_refinement(Spectral::MortarSize::LowerHalf);
  test_h_refinement(Spectral::MortarSize::UpperHalf);
}
