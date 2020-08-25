// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

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
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | element_id;
    p | is_projected;
    p | is_oriented;
  }
};

template <size_t Dim>
struct MortarDataRecorder {
  void local_insert(TemporalId temporal_id,
                    TestBoundaryData<Dim> data) noexcept {
    temporal_id_ = temporal_id;
    local_data_ = std::move(data);
  }

  void remote_insert(TemporalId temporal_id,
                     TestBoundaryData<Dim> data) noexcept {
    received_data.emplace_back(temporal_id, std::move(data));
  }

  const TestBoundaryData<Dim>& local_data(const TemporalId& temporal_id) const
      noexcept {
    CHECK(temporal_id == temporal_id_);
    return local_data_;
  }

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | temporal_id_;
    p | local_data_;
    p | received_data;
  }

  // Only recording remote data
  TemporalId temporal_id_ = 0;
  TestBoundaryData<Dim> local_data_{};
  std::vector<std::pair<TemporalId, TestBoundaryData<Dim>>> received_data{};
};

template <size_t Dim>
struct MortarDataTag : db::SimpleTag {
  using type = MortarDataRecorder<Dim>;
};

template <size_t Dim>
struct DgBoundaryScheme {
  static constexpr size_t volume_dim = Dim;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = ::Tags::Next<TemporalIdTag>;
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
struct lts_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<>;

  using simple_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, TemporalIdTag,
                 ::Tags::Next<TemporalIdTag>, domain::Tags::Mesh<Dim>,
                 domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>>;
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
      tmpl::list<::Tags::Mortars<::Tags::Next<TemporalIdTag>, Dim>,
                 ::Tags::Mortars<MortarDataTag<Dim>, Dim>,
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
              dg::Actions::InitializeMortars<DgBoundaryScheme<Dim>, false>,
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme<Dim>, domain::Tags::InternalDirections<Dim>>,
              dg::Actions::SendDataForFluxes<DgBoundaryScheme<Dim>>,
              dg::Actions::ReceiveDataForFluxes<DgBoundaryScheme<Dim>>,
              Initialization::Actions::RemoveOptionsAndTerminatePhase>>>;
};

template <size_t Dim>
struct LtsMetavariables {
  using component_list = tmpl::list<lts_component<Dim, LtsMetavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

// Inserts the new neighbor element into the MockRuntimeSystem
template <typename Metavariables>
void insert_neighbor(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<Metavariables>*>
        runner,
    const Element<2>& element, const int start) noexcept {
  using metavariables = Metavariables;
  using my_component = lts_component<2, metavariables>;
  const Mesh<2> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const std::vector<std::array<size_t, 2>> initial_extents{
      {{2, 2}}, {{2, 2}}, {{2, 2}}};

  ElementMap<2, Frame::Inertial> map(
      element.id(),
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}));

  ActionTesting::emplace_component_and_initialize<my_component>(
      runner, element.id(),
      {initial_extents, start, start, mesh, element, std::move(map)});
}

// We have to increment the "next" time on self after initialization, since we
// had it set to the "current" time during initialization so it got picked up on
// mortars by `InitializeMortars`
template <typename Component, typename Runner, typename ElementIdType>
void set_next_temporal_id(const gsl::not_null<Runner*> runner,
                          const ElementIdType& element_id,
                          const TemporalId& time) noexcept {
  auto& box =
      ActionTesting::get_databox<Component, typename Component::db_tags_list>(
          runner, element_id);
  db::mutate<::Tags::Next<TemporalIdTag>>(
      make_not_null(&box),
      [&time](const gsl::not_null<TemporalId*> next_temporal_id) {
        *next_temporal_id = time;
      });
}

// Update the time and flux, then send the data
void send_from_neighbor(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<LtsMetavariables<2>>*>
        runner,
    const Element<2>& element, const int start, const int end) noexcept {
  using metavariables = LtsMetavariables<2>;
  using my_component = lts_component<2, metavariables>;
  db::mutate<TemporalIdTag, Tags::Next<TemporalIdTag>>(
      make_not_null(
          &ActionTesting::get_databox<my_component,
                                      typename my_component::db_tags_list>(
              runner, element.id())),
      [start, end](auto tstart, auto tend) {
        *tstart = start;
        *tend = end;
      });

  runner->force_next_action_to_be<
      my_component,
      dg::Actions::CollectDataForFluxes<DgBoundaryScheme<2>,
                                        domain::Tags::InternalDirections<2>>>(
      element.id());
  runner->next_action<my_component>(element.id());
  runner->next_action<my_component>(element.id());
}

// Sends the left steps, then the right steps.  The step must not be
// ready until after the last send.
void run_lts_case(const int self_step_end, const std::vector<int>& left_steps,
                  const std::vector<int>& right_steps) noexcept {
  using metavariables = LtsMetavariables<2>;
  using my_component = lts_component<2, metavariables>;
  using all_mortar_data_tag = ::Tags::Mortars<MortarDataTag<2>, 2>;
  using mortar_next_temporal_ids_tag =
      ::Tags::Mortars<::Tags::Next<TemporalIdTag>, 2>;

  const int self_step_start = 0;  // We always start at 0

  const ElementId<2> left_id(0);
  const ElementId<2> self_id(1);
  const ElementId<2> right_id(2);

  using MortarId = std::pair<Direction<2>, ElementId<2>>;
  const MortarId left_mortar_id(Direction<2>::lower_xi(), left_id);
  const MortarId right_mortar_id(Direction<2>::upper_xi(), right_id);

  const Mesh<2> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const std::vector<std::array<size_t, 2>> initial_extents{
      {{2, 2}}, {{2, 2}}, {{2, 2}}};

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};

  PUPable_reg(
      SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                       domain::CoordinateMaps::Identity<2>>));
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {initial_extents, self_step_start, self_step_start, mesh,
       Element<2>{self_id,
                  {{Direction<2>::lower_xi(), {{left_id}, {}}},
                   {Direction<2>::upper_xi(), {{right_id}, {}}}}},
       ElementMap<2, Frame::Inertial>{
           self_id,
           domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
               domain::CoordinateMaps::Identity<2>{})}});

  // Insert the left and right neighbor
  const Element<2> left_element(left_id,
                                {{Direction<2>::upper_xi(), {{self_id}, {}}}});
  const Element<2> right_element(right_id,
                                 {{Direction<2>::lower_xi(), {{self_id}, {}}}});

  insert_neighbor(make_not_null(&runner), left_element, 0);
  insert_neighbor(make_not_null(&runner), right_element, 0);
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<my_component, tag>(runner, self_id);
  };

  // InitializeMortars on self
  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);
  set_next_temporal_id<my_component>(make_not_null(&runner), self_id,
                                     self_step_end);
  db::mutate<mortar_next_temporal_ids_tag>(
      make_not_null(
          &ActionTesting::get_databox<my_component,
                                      typename my_component::db_tags_list>(
              make_not_null(&runner), self_id)),
      [&left_mortar_id, &right_mortar_id, &left_steps,
       &right_steps](auto mortar_next_temporal_ids) {
        mortar_next_temporal_ids->at(left_mortar_id) = left_steps.front();
        mortar_next_temporal_ids->at(right_mortar_id) = right_steps.front();
      });
  // InitializeMortars on neighbors
  ActionTesting::next_action<my_component>(make_not_null(&runner), left_id);
  set_next_temporal_id<my_component>(make_not_null(&runner), left_id, 1);
  ActionTesting::next_action<my_component>(make_not_null(&runner), right_id);
  set_next_temporal_id<my_component>(make_not_null(&runner), right_id, 1);

  // CollectDataForFluxes on self
  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);
  // SendDataForFluxes on self
  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);

  std::vector<int> relevant_left_steps{left_steps.front()};
  for (size_t step = 1; step < left_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<my_component>(self_id));
    send_from_neighbor(&runner, left_element, left_steps[step - 1],
                       left_steps[step]);
    if (left_steps[step - 1] < self_step_end) {
      relevant_left_steps.push_back(left_steps[step]);
    }
  }
  std::vector<int> relevant_right_steps{right_steps.front()};
  for (size_t step = 1; step < right_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<my_component>(self_id));
    send_from_neighbor(&runner, right_element, right_steps[step - 1],
                       right_steps[step]);
    if (right_steps[step - 1] < self_step_end) {
      relevant_right_steps.push_back(right_steps[step]);
    }
  }
  REQUIRE(ActionTesting::is_ready<my_component>(runner, self_id));
  // ReceiveDataForFluxes
  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);

  CHECK(get_tag(mortar_next_temporal_ids_tag{}) ==
        typename mortar_next_temporal_ids_tag::type{
            {left_mortar_id, relevant_left_steps.back()},
            {right_mortar_id, relevant_right_steps.back()}});

  const auto& recorders = get_tag(all_mortar_data_tag{});
  const auto check_data = [](const auto& recorder,
                             const std::vector<int>& steps) noexcept {
    const auto& received_data = recorder.received_data;
    CHECK(received_data.size() == steps.size() - 1);

    for (size_t step = 0; step < received_data.size(); ++step) {
      CHECK(received_data[step].first == steps[step]);
    }
  };
  check_data(recorders.at(left_mortar_id), relevant_left_steps);
  check_data(recorders.at(right_mortar_id), relevant_right_steps);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication.lts",
                  "[Unit][NumericalAlgorithms][Actions]") {
  // Global step 0 -> 1
  run_lts_case(1, {0, 1}, {0, 1});
  // Global step 0 -> 1 with left element taking second step
  run_lts_case(1, {0, 1, 2}, {0, 1});
  // Neighbors stepping past element
  run_lts_case(1, {0, 2}, {0, 1});
  run_lts_case(1, {0, 1}, {0, 2});
  run_lts_case(1, {0, 2}, {0, 2});
  // No receives from one or both sides (because one or more neighbors
  // have already made it to the desired time)
  run_lts_case(1, {1}, {1});
  run_lts_case(1, {1}, {2});
  run_lts_case(1, {2}, {1});
  run_lts_case(1, {2}, {2});
  run_lts_case(1, {0, 1}, {1});
  run_lts_case(1, {0, 1}, {2});
  run_lts_case(1, {0, 2}, {1});
  run_lts_case(1, {0, 2}, {2});
  run_lts_case(1, {1}, {0, 1});
  run_lts_case(1, {1}, {0, 2});
  run_lts_case(1, {2}, {0, 1});
  run_lts_case(1, {2}, {0, 2});
  // Several steps to receive
  run_lts_case(3, {0, 1, 3, 4}, {0, 1, 2, 4});
}
