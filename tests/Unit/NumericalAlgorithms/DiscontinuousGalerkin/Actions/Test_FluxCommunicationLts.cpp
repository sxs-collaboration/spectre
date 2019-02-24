// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DirectionMap.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>
// IWYU pragma: no_include <boost/variant/get.hpp>

// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"  // for Variables
// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; }
  using type = int;
};

struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

struct OtherData : db::SimpleTag {
  static std::string name() noexcept { return "OtherData"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim>
class NumericalFlux {
 public:
  struct ExtraData : db::SimpleTag {
    static std::string name() noexcept { return "ExtraTag"; }
    using type = tnsr::I<DataVector, 1>;
  };

  using package_tags = tmpl::list<ExtraData, Var>;
  // This is a silly set of things to request, but it tests not
  // requesting the evolved variables and requesting multiple other
  // things.
  using argument_tags =
      tmpl::list<Tags::NormalDotFlux<Var>, OtherData,
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>;
  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& var_flux,
                    const Scalar<DataVector>& other_data,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) const noexcept {
    get(get<Var>(*packaged_data)) = 10. * get(var_flux);
    get<0>(get<ExtraData>(*packaged_data)) =
        get(other_data) + 2. * get<0>(interface_unit_normal) +
        3. * get<1>(interface_unit_normal);
  }

  // void operator()(...) is unused

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct NumericalFluxTag {
  using type = NumericalFlux<Dim>;
};

template <size_t Dim>
struct System {
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim, typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;
template <size_t Dim, typename Tag>
using interface_compute_tag =
    Tags::InterfaceComputeItem<Tags::InternalDirections<Dim>, Tag>;

template <typename FluxCommTypes>
using LocalData = typename FluxCommTypes::LocalData;
template <typename FluxCommTypes>
using PackagedData = typename FluxCommTypes::PackagedData;
template <size_t Dim, typename FluxCommTypes>
using normal_dot_fluxes_tag =
    interface_tag<Dim, typename FluxCommTypes::normal_dot_fluxes_tag>;

template <size_t Dim>
using other_data_tag =
    interface_tag<Dim, Tags::Variables<tmpl::list<OtherData>>>;
template <size_t Dim>
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, Dim>;
template <size_t Dim>
using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>;
template <size_t Dim>
using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>;

struct DataRecorder;

struct DataRecorderTag : db::SimpleTag {
  static std::string name() { return "DataRecorderTag"; }
  using type = DataRecorder;
};

struct MortarRecorderTag : Tags::VariablesBoundaryData,
                           Tags::Mortars<DataRecorderTag, 2> {};

template <size_t Dim, typename MV>
struct lts_component {
  using metavariables = MV;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<NumericalFluxTag<Dim>>;
  using action_list = tmpl::list<dg::Actions::SendDataForFluxes<MV>,
                                 dg::Actions::ReceiveDataForFluxes<MV>>;
  using flux_comm_types = dg::FluxCommunicationTypes<MV>;

  using simple_tags = db::AddSimpleTags<
      TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>, Tags::Element<2>,
      Tags::ElementMap<2>, normal_dot_fluxes_tag<2, flux_comm_types>,
      other_data_tag<2>, mortar_meshes_tag<2>, mortar_sizes_tag<2>,
      MortarRecorderTag, mortar_next_temporal_ids_tag<2>>;

  using compute_tags = db::AddComputeTags<
      Tags::InternalDirections<Dim>,
      interface_compute_tag<Dim, Tags::Direction<Dim>>,
      interface_compute_tag<Dim, Tags::InterfaceMesh<Dim>>,
      interface_compute_tag<Dim, Tags::UnnormalizedFaceNormal<Dim>>,
      interface_compute_tag<
          Dim, Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
      interface_compute_tag<
          Dim, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

template <size_t Dim>
struct LtsMetavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<lts_component<Dim, LtsMetavariables>>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag<Dim>;
};

template <size_t Dim>
using flux_comm_types = dg::FluxCommunicationTypes<LtsMetavariables<Dim>>;

template <typename Component>
using compute_items = typename Component::compute_tags;

struct DataRecorder {
  // Only called on the sending sides, which are not interesting here.
  void local_insert(int /*temporal_id*/,
                    const LocalData<flux_comm_types<2>>& /*data*/) noexcept {}

  void remote_insert(int temporal_id,
                     PackagedData<flux_comm_types<2>> data) noexcept {
    received_data.emplace_back(temporal_id, std::move(data));
  }

  std::vector<std::pair<int, PackagedData<flux_comm_types<2>>>> received_data{};
};

// Inserts the new neighbor element into the MockRuntimeSystem
template <typename LocalAlg>
void insert_neighbor(const gsl::not_null<LocalAlg*> local_alg,
                     const Element<2>& element, const int start, const int end,
                     const double n_dot_f) noexcept {
  using metavariables = LtsMetavariables<2>;
  using my_component = lts_component<2, metavariables>;
  const Direction<2>& send_direction = element.neighbors().begin()->first;
  const ElementId<2>& receiver_id =
      *element.neighbors().begin()->second.begin();

  const Mesh<2> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  ElementMap<2, Frame::Inertial> map(
      element.id(),
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::Identity<2>{}));

  db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>> fluxes;
  fluxes[send_direction].initialize(2, n_dot_f);

  db::item_type<other_data_tag<2>> other_data;
  other_data[send_direction].initialize(2, 0.);

  db::item_type<mortar_meshes_tag<2>> mortar_meshes;
  mortar_meshes.insert({{send_direction, receiver_id}, mesh.slice_away(0)});

  db::item_type<mortar_sizes_tag<2>> mortar_sizes;
  mortar_sizes.insert(
      {{send_direction, receiver_id}, {{Spectral::MortarSize::Full}}});

  db::item_type<MortarRecorderTag> recorders;
  recorders.insert({{send_direction, receiver_id}, {}});

  auto box = db::create<
      db::AddSimpleTags<
          TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>, Tags::Element<2>,
          Tags::ElementMap<2>, normal_dot_fluxes_tag<2, flux_comm_types<2>>,
          other_data_tag<2>, mortar_meshes_tag<2>, mortar_sizes_tag<2>,
          MortarRecorderTag, mortar_next_temporal_ids_tag<2>>,
      compute_items<my_component>>(
      start, end, mesh, element, std::move(map), std::move(fluxes),
      std::move(other_data), std::move(mortar_meshes), std::move(mortar_sizes),
      std::move(recorders), db::item_type<mortar_next_temporal_ids_tag<2>>{});

  local_alg->emplace(element.id(), std::move(box));
}

// Update the time and flux, then send the data
void send_from_neighbor(
    const gsl::not_null<ActionTesting::MockRuntimeSystem<LtsMetavariables<2>>*>
        runner,
    const Element<2>& element, const int start, const int end,
    const double n_dot_f) noexcept {
  using metavariables = LtsMetavariables<2>;
  using my_component = lts_component<2, metavariables>;
  using initial_databox_type = typename my_component::initial_databox;
  const Direction<2>& send_direction = element.neighbors().begin()->first;

  db::mutate<TemporalId, Tags::Next<TemporalId>,
             normal_dot_fluxes_tag<2, flux_comm_types<2>>, other_data_tag<2>>(
      make_not_null(&runner->algorithms<my_component>()
                         .at(element.id())
                         .get_databox<initial_databox_type>()),
      [&send_direction, n_dot_f, start, end](auto tstart, auto tend,
                                             auto fluxes, auto other_data) {
        *tstart = start;
        *tend = end;
        fluxes->operator[](send_direction).initialize(2, n_dot_f);
        other_data->operator[](send_direction).initialize(2, 0.);
      });

  runner->force_next_action_to_be<
      my_component, dg::Actions::SendDataForFluxes<metavariables>>(
      element.id());
  runner->next_action<my_component>(element.id());
}

// Sends the left steps, then the right steps.  The step must not be
// ready until after the last send.
void run_lts_case(const int self_step_end, const std::vector<int>& left_steps,
                  const std::vector<int>& right_steps) noexcept {
  using metavariables = LtsMetavariables<2>;
  using my_component = lts_component<2, metavariables>;
  using initial_databox_type = typename my_component::initial_databox;

  const int self_step_start = 0;  // We always start at 0

  const ElementId<2> left_id(0);
  const ElementId<2> self_id(1);
  const ElementId<2> right_id(2);

  using MortarId = std::pair<Direction<2>, ElementId<2>>;
  const MortarId left_mortar_id(Direction<2>::lower_xi(), left_id);
  const MortarId right_mortar_id(Direction<2>::upper_xi(), right_id);

  db::item_type<MortarRecorderTag> initial_recorders;
  initial_recorders.insert({{Direction<2>::lower_xi(), left_id}, {}});
  initial_recorders.insert({{Direction<2>::upper_xi(), right_id}, {}});
  db::item_type<mortar_next_temporal_ids_tag<2>> initial_mortar_temporal_ids{
      {left_mortar_id, left_steps.front()},
      {right_mortar_id, right_steps.front()}};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<my_component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(
          self_id,
          ActionTesting::MockDistributedObject<my_component>{db::create<
              db::AddSimpleTags<
                  TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>,
                  Tags::Element<2>, Tags::ElementMap<2>,
                  normal_dot_fluxes_tag<2, flux_comm_types<2>>,
                  other_data_tag<2>, mortar_meshes_tag<2>, mortar_sizes_tag<2>,
                  MortarRecorderTag, mortar_next_temporal_ids_tag<2>>,
              typename my_component::compute_tags>(
              self_step_start, self_step_end,
              Mesh<2>{2, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto},
              Element<2>{},
              ElementMap<2, Frame::Inertial>{
                  self_id, domain::make_coordinate_map_base<Frame::Logical,
                                                            Frame::Inertial>(
                               domain::CoordinateMaps::Identity<2>{})},
              db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>{},
              db::item_type<other_data_tag<2>>{},
              db::item_type<mortar_meshes_tag<2>>{},
              db::item_type<mortar_sizes_tag<2>>{},
              std::move(initial_recorders),
              std::move(initial_mortar_temporal_ids))});

  // Insert the left and right neighbor
  const Element<2> left_element(left_id,
                                {{Direction<2>::upper_xi(), {{self_id}, {}}}});
  const Element<2> right_element(right_id,
                                 {{Direction<2>::lower_xi(), {{self_id}, {}}}});

  insert_neighbor(
      make_not_null(&tuples::get<MockDistributedObjectsTag>(dist_objects)),
      left_element, 0, 1, 0.0);
  insert_neighbor(
      make_not_null(&tuples::get<MockDistributedObjectsTag>(dist_objects)),
      right_element, 0, 1, 0.0);

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {NumericalFlux<2>{}}, std::move(dist_objects)};

  runner.next_action<my_component>(self_id);  // SendDataForFluxes

  std::vector<int> relevant_left_steps{left_steps.front()};
  for (size_t step = 1; step < left_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<my_component>(self_id));
    send_from_neighbor(&runner, left_element, left_steps[step - 1],
                       left_steps[step], step);
    if (left_steps[step - 1] < self_step_end) {
      relevant_left_steps.push_back(left_steps[step]);
    }
  }
  std::vector<int> relevant_right_steps{right_steps.front()};
  for (size_t step = 1; step < right_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<my_component>(self_id));
    send_from_neighbor(&runner, right_element, right_steps[step - 1],
                       right_steps[step], step);
    if (right_steps[step - 1] < self_step_end) {
      relevant_right_steps.push_back(right_steps[step]);
    }
  }
  REQUIRE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(self_id);  // ReceiveDataForFluxes

  auto& box = runner.algorithms<my_component>()
                  .at(self_id)
                  .get_databox<initial_databox_type>();

  CHECK(db::get<mortar_next_temporal_ids_tag<2>>(box) ==
        db::item_type<mortar_next_temporal_ids_tag<2>>{
            {left_mortar_id, relevant_left_steps.back()},
            {right_mortar_id, relevant_right_steps.back()}});

  const auto& recorders = db::get<MortarRecorderTag>(box);
  const auto check_data = [](const auto& recorder,
                             const std::vector<int>& steps) noexcept {
    const auto& received_data = recorder.received_data;
    CHECK(received_data.size() == steps.size() - 1);

    for (size_t step = 0; step < received_data.size(); ++step) {
      CHECK(received_data[step].first == steps[step]);
      CHECK(get(get<Var>(received_data[step].second)) ==
            DataVector(2, 10. * (step + 1)));
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
