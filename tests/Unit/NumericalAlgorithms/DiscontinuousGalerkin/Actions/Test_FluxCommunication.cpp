// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>
#include <cstddef>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <map>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

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
  static constexpr bool should_be_sliced_to_boundary = false;
};

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
                 Tags::Normalized<Tags::UnnormalizedFaceNormal<2>>>;
  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& var_flux,
                    const Scalar<DataVector>& other_data,
                    const tnsr::i<DataVector, 2, Frame::Inertial>&
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

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

struct Metavariables;
using send_data_for_fluxes = dg::Actions::SendDataForFluxes<Metavariables>;
using receive_data_for_fluxes =
    dg::Actions::ReceiveDataForFluxes<Metavariables>;

using component =
    ActionTesting::MockArrayComponent<Metavariables, ElementIndex<2>,
                                      tmpl::list<NumericalFluxTag>,
                                      tmpl::list<receive_data_for_fluxes>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag;
};

template <typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<2>, Tag>;

using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
using LocalData = typename flux_comm_types::LocalData;
using PackagedData = typename flux_comm_types::PackagedData;
using MagnitudeOfFaceNormal = typename flux_comm_types::MagnitudeOfFaceNormal;
using normal_dot_fluxes_tag =
    interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>;

using fluxes_tag = typename flux_comm_types::FluxesTag;

using other_data_tag = interface_tag<Tags::Variables<tmpl::list<OtherData>>>;
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, 2>;

using compute_items = db::AddComputeTags<
    Tags::InternalDirections<2>, interface_tag<Tags::Direction<2>>,
    interface_tag<Tags::Mesh<1>>,
    interface_tag<Tags::UnnormalizedFaceNormal<2>>,
    interface_tag<Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>,
    interface_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<2>>>>;

Scalar<DataVector> reverse(Scalar<DataVector> x) noexcept {
  std::reverse(get(x).begin(), get(x).end());
  return x;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{NumericalFlux{}}};

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

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

  // OrientationMap from block 1 to block 0
  const OrientationMap<2> block_orientation(
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
      {{Direction<2>::lower_eta(), Direction<2>::lower_xi()}});

  // Since we're lazy and use the same map for both blocks (the
  // actions are only sensitive to the ElementMap, which does differ),
  // we need to make the xi and eta maps line up along the block
  // interface.
  const CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const CoordinateMaps::Affine eta_map{-1., 1., 7., 3.};

  const auto coordmap =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>(xi_map,
                                                                 eta_map));

  const auto neighbor_directions = {Direction<2>::lower_xi(),
                                    Direction<2>::upper_xi(),
                                    Direction<2>::upper_eta()};
  const auto neighbor_mortar_ids = {
      std::make_pair(Direction<2>::lower_xi(), west_id),
      std::make_pair(Direction<2>::upper_xi(), east_id),
      std::make_pair(Direction<2>::upper_eta(), south_id)};
  const struct {
    std::unordered_map<Direction<2>, Scalar<DataVector>> fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>> other_data;
    std::unordered_map<Direction<2>, Scalar<DataVector>> remote_fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>> remote_other_data;
  } data{
      {{Direction<2>::lower_xi(), Scalar<DataVector>{{{{1., 2., 3.}}}}},
       {Direction<2>::upper_xi(), Scalar<DataVector>{{{{4., 5., 6.}}}}},
       {Direction<2>::upper_eta(), Scalar<DataVector>{{{{7., 8., 9.}}}}}},
      {{Direction<2>::lower_xi(), Scalar<DataVector>{{{{10., 11., 12.}}}}},
       {Direction<2>::upper_xi(), Scalar<DataVector>{{{{13., 14., 15.}}}}},
       {Direction<2>::upper_eta(), Scalar<DataVector>{{{{16., 17., 18.}}}}}},
      {{Direction<2>::lower_xi(), Scalar<DataVector>{{{{19., 20., 21.}}}}},
       {Direction<2>::upper_xi(), Scalar<DataVector>{{{{22., 23., 24.}}}}},
       {Direction<2>::upper_eta(), Scalar<DataVector>{{{{25., 26., 27.}}}}}},
      {{Direction<2>::lower_xi(), Scalar<DataVector>{{{{28., 29., 30.}}}}},
       {Direction<2>::upper_xi(), Scalar<DataVector>{{{{31., 32., 33.}}}}},
       {Direction<2>::upper_eta(), Scalar<DataVector>{{{{34., 35., 36.}}}}}}};

  auto start_box = [
    &mesh, &self_id, &west_id, &east_id, &south_id, &block_orientation,
    &coordmap, &neighbor_directions, &neighbor_mortar_ids, &data
  ]() noexcept {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes;
    for (const auto& direction : neighbor_directions) {
      auto& flux_vars = normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<Var>>(flux_vars) = data.fluxes.at(direction);
    }

    db::item_type<other_data_tag> other_data;
    for (const auto& direction : neighbor_directions) {
      auto& other_data_vars = other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) = data.other_data.at(direction);
    }

    db::item_type<mortar_data_tag> mortar_history{};
    db::item_type<mortar_next_temporal_ids_tag> mortar_next_temporal_ids{};
    for (const auto& mortar_id : neighbor_mortar_ids) {
      mortar_history.insert({mortar_id, {}});
      mortar_next_temporal_ids.insert({mortar_id, 0});
    }

    return db::create<
        db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>,
                          Tags::Element<2>, Tags::ElementMap<2>,
                          normal_dot_fluxes_tag, other_data_tag,
                          mortar_data_tag, mortar_next_temporal_ids_tag>,
        compute_items>(0, 1, mesh, element, std::move(map),
                       std::move(normal_dot_fluxes), std::move(other_data),
                       std::move(mortar_history),
                       std::move(mortar_next_temporal_ids));
  }();

  auto sent_box = std::get<0>(
      runner.apply<component, send_data_for_fluxes>(start_box, self_id));

  // Here, we just check that messages are sent to the correct places.
  // We will check the received values on the central element later.
  {
    CHECK(runner.nonempty_inboxes<component, fluxes_tag>() ==
          std::unordered_set<ElementIndex<2>>{west_id, east_id, south_id});
    const auto check_sent_data = [&runner, &self_id](
        const ElementId<2>& id, const Direction<2>& direction) noexcept {
      const auto& inboxes = runner.inboxes<component>();
      const auto& flux_inbox = tuples::get<fluxes_tag>(inboxes.at(id));
      CHECK(flux_inbox.size() == 1);
      CHECK(flux_inbox.count(0) == 1);
      const auto& flux_inbox_at_time = flux_inbox.at(0);
      CHECK(flux_inbox_at_time.size() == 1);
      CHECK(flux_inbox_at_time.count({direction, self_id}) == 1);
    };
    check_sent_data(west_id, Direction<2>::lower_eta());
    check_sent_data(east_id, Direction<2>::lower_xi());
    check_sent_data(south_id, Direction<2>::lower_eta());
  }

  // Now check ReceiveDataForFluxes
  const auto send_data = [&mesh, &runner, &self_id, &coordmap](
      const ElementId<2>& id, const Direction<2>& direction,
      const OrientationMap<2>& orientation,
      const Scalar<DataVector>& normal_dot_fluxes,
      const Scalar<DataVector>& other_data) noexcept {
    const Element<2> element(id, {{direction, {{self_id}, orientation}}});
    auto map = ElementMap<2, Frame::Inertial>(id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes_map{};
    normal_dot_fluxes_map[direction].initialize(get(normal_dot_fluxes).size());
    get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes_map[direction]) =
        normal_dot_fluxes;

    db::item_type<other_data_tag> other_data_map{};
    other_data_map[direction].initialize(get(other_data).size());
    get<OtherData>(other_data_map[direction]) = other_data;

    db::item_type<mortar_data_tag> mortar_history{};
    mortar_history[std::make_pair(direction, self_id)];

    auto box = db::create<
        db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>,
                          Tags::Mesh<2>, Tags::Element<2>, Tags::ElementMap<2>,
                          normal_dot_fluxes_tag, other_data_tag,
                          mortar_data_tag>,
        compute_items>(0, 1, mesh, element, std::move(map),
                       std::move(normal_dot_fluxes_map),
                       std::move(other_data_map), std::move(mortar_history));

    runner.apply<component, send_data_for_fluxes>(box, id);
  };

  CHECK_FALSE(
      (runner.is_ready<component, receive_data_for_fluxes>(sent_box, self_id)));
  send_data(south_id, Direction<2>::lower_eta(), {},
            data.remote_fluxes.at(Direction<2>::upper_eta()),
            data.remote_other_data.at(Direction<2>::upper_eta()));
  CHECK_FALSE(
      (runner.is_ready<component, receive_data_for_fluxes>(sent_box, self_id)));
  send_data(east_id, Direction<2>::lower_xi(), {},
            data.remote_fluxes.at(Direction<2>::upper_xi()),
            data.remote_other_data.at(Direction<2>::upper_xi()));
  CHECK_FALSE(
      (runner.is_ready<component, receive_data_for_fluxes>(sent_box, self_id)));
  send_data(west_id, Direction<2>::lower_eta(), block_orientation.inverse_map(),
            data.remote_fluxes.at(Direction<2>::lower_xi()),
            data.remote_other_data.at(Direction<2>::lower_xi()));
  CHECK(runner.is_ready<component, receive_data_for_fluxes>(sent_box, self_id));

  auto received_box = std::get<0>(
      runner.apply<component, receive_data_for_fluxes>(sent_box, self_id));

  CHECK(tuples::get<fluxes_tag>(runner.inboxes<component>()[self_id]).empty());

  for (const auto& mortar_id : neighbor_mortar_ids) {
    CHECK(db::get<mortar_next_temporal_ids_tag>(received_box).at(mortar_id) ==
          1);
  };

  db::mutate<mortar_data_tag>(make_not_null(&received_box), [
    &west_id, &east_id, &south_id, &data
  ](const gsl::not_null<db::item_type<mortar_data_tag>*>
        mortar_history) noexcept {
    CHECK(mortar_history->size() == 3);
    const auto check_mortar = [&mortar_history](
        const std::pair<Direction<2>, ElementId<2>>& mortar_id,
        const Scalar<DataVector>& local_flux,
        const Scalar<DataVector>& remote_flux,
        const Scalar<DataVector>& local_other,
        const Scalar<DataVector>& remote_other,
        const tnsr::i<DataVector, 2>& local_normal,
        const tnsr::i<DataVector, 2>& remote_normal) noexcept {
      LocalData local_data(3);
      get<Tags::NormalDotFlux<Var>>(local_data) = local_flux;
      get<MagnitudeOfFaceNormal>(local_data) = magnitude(local_normal);
      auto normalized_local_normal = local_normal;
      for (auto& x : normalized_local_normal) {
        x /= get(get<MagnitudeOfFaceNormal>(local_data));
      }
      PackagedData local_packaged(3);
      NumericalFlux{}.package_data(&local_packaged, local_flux,
                                   local_other, normalized_local_normal);
      local_data.assign_subset(local_packaged);

      const auto magnitude_remote_normal = magnitude(remote_normal);
      auto normalized_remote_normal = remote_normal;
      for (auto& x : normalized_remote_normal) {
        x /= get(magnitude_remote_normal);
      }
      PackagedData remote_packaged(3);
      NumericalFlux{}.package_data(&remote_packaged, remote_flux,
                                   remote_other, normalized_remote_normal);
      // Cannot be inlined because of CHECK implementation.
      const auto expected =
          std::make_pair(std::move(local_data), std::move(remote_packaged));
      CHECK(mortar_history->at(mortar_id).extract() == expected);
    };

    // Remote side is inverted
    check_mortar(
        std::make_pair(Direction<2>::lower_xi(), west_id),
        data.fluxes.at(Direction<2>::lower_xi()),
        reverse(data.remote_fluxes.at(Direction<2>::lower_xi())),
        data.other_data.at(Direction<2>::lower_xi()),
        reverse(data.remote_other_data.at(Direction<2>::lower_xi())),
        tnsr::i<DataVector, 2>{{{DataVector{3, -2.0}, DataVector{3, 0.0}}}},
        tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, 0.5}}}});
    check_mortar(
        std::make_pair(Direction<2>::upper_xi(), east_id),
        data.fluxes.at(Direction<2>::upper_xi()),
        data.remote_fluxes.at(Direction<2>::upper_xi()),
        data.other_data.at(Direction<2>::upper_xi()),
        data.remote_other_data.at(Direction<2>::upper_xi()),
        tnsr::i<DataVector, 2>{{{DataVector{3, 2.0}, DataVector{3, 0.0}}}},
        tnsr::i<DataVector, 2>{{{DataVector{3, -2.0}, DataVector{3, 0.0}}}});
    check_mortar(
        std::make_pair(Direction<2>::upper_eta(), south_id),
        data.fluxes.at(Direction<2>::upper_eta()),
        data.remote_fluxes.at(Direction<2>::upper_eta()),
        data.other_data.at(Direction<2>::upper_eta()),
        data.remote_other_data.at(Direction<2>::upper_eta()),
        tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, -1.0}}}},
        tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, 1.0}}}});
  });
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.NoNeighbors",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{NumericalFlux{}}};

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  const Element<2> element(self_id, {});

  auto map = ElementMap<2, Frame::Inertial>(
      self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                   CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>(
                       {-1., 1., 3., 7.}, {-1., 1., -2., 4.})));

  auto start_box = db::create<
      db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>,
                        Tags::Element<2>, Tags::ElementMap<2>,
                        normal_dot_fluxes_tag, other_data_tag, mortar_data_tag,
                        mortar_next_temporal_ids_tag>,
      compute_items>(0, 1, mesh, element, std::move(map),
                     db::item_type<normal_dot_fluxes_tag>{},
                     db::item_type<other_data_tag>{},
                     db::item_type<mortar_data_tag>{},
                     db::item_type<mortar_next_temporal_ids_tag>{});

  auto sent_box = std::get<0>(
      runner.apply<component, send_data_for_fluxes>(start_box, self_id));

  CHECK(db::get<mortar_data_tag>(sent_box).empty());
  CHECK(runner.nonempty_inboxes<component, fluxes_tag>().empty());

  CHECK(runner.is_ready<component, receive_data_for_fluxes>(sent_box, self_id));

  const auto received_box = std::get<0>(
      runner.apply<component, receive_data_for_fluxes>(sent_box, self_id));

  CHECK(db::get<mortar_data_tag>(received_box).empty());
  CHECK(runner.nonempty_inboxes<component, fluxes_tag>().empty());
}

namespace {
struct DataRecorder {
  // Only called on the sending sides, which are not interesting here.
  void local_insert(int /*temporal_id*/, const LocalData& /*data*/) noexcept {}

  void remote_insert(int temporal_id, PackagedData data) noexcept {
    received_data.emplace_back(temporal_id, std::move(data));
  }

  std::vector<std::pair<int, PackagedData>> received_data{};
};

struct DataRecorderTag : db::SimpleTag {
  static std::string name() { return "DataRecorderTag"; }
  using type = DataRecorder;
};

struct MortarRecorderTag : Tags::VariablesBoundaryData,
                           Tags::Mortars<DataRecorderTag, 2> {};

void send_from_neighbor(
    const gsl::not_null<ActionTesting::ActionRunner<Metavariables>*> runner,
    const Element<2>& element, const int start, const int end,
    const double n_dot_f) noexcept {
  const Direction<2>& send_direction = element.neighbors().begin()->first;
  const ElementId<2>& receiver_id =
      *element.neighbors().begin()->second.begin();

  ElementMap<2, Frame::Inertial> map(
      element.id(), make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                        CoordinateMaps::Identity<2>{}));

  db::item_type<normal_dot_fluxes_tag> fluxes;
  fluxes[send_direction].initialize(2, n_dot_f);

  db::item_type<other_data_tag> other_data;
  other_data[send_direction].initialize(2, 0.);

  db::item_type<MortarRecorderTag> recorders;
  recorders.insert({{send_direction, receiver_id}, {}});

  const Mesh<2> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  auto box =
      db::create<db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>,
                                   Tags::Mesh<2>, Tags::Element<2>,
                                   Tags::ElementMap<2>, normal_dot_fluxes_tag,
                                   other_data_tag, MortarRecorderTag>,
                 compute_items>(start, end, mesh, element,
                                std::move(map), std::move(fluxes),
                                std::move(other_data), std::move(recorders));

  runner->apply<component, send_data_for_fluxes>(box, element.id());
}

// Sends the left steps, then the right steps.  The step must not be
// ready until after the last send.
void run_lts_case(const int self_step_end, const std::vector<int>& left_steps,
                  const std::vector<int>& right_steps) noexcept {
  ActionTesting::ActionRunner<Metavariables> runner{{NumericalFlux{}}};

  const ElementId<2> left_id(0);
  const ElementId<2> self_id(1);
  const ElementId<2> right_id(2);

  using MortarId = std::pair<Direction<2>, ElementId<2>>;
  const MortarId left_mortar_id(Direction<2>::lower_xi(), left_id);
  const MortarId right_mortar_id(Direction<2>::upper_xi(), right_id);

  db::item_type<MortarRecorderTag> initial_recorders;
  initial_recorders.insert({{Direction<2>::lower_xi(), left_id}, {}});
  initial_recorders.insert({{Direction<2>::upper_xi(), right_id}, {}});
  db::item_type<mortar_next_temporal_ids_tag> initial_mortar_temporal_ids{
      {left_mortar_id, left_steps.front()},
      {right_mortar_id, right_steps.front()}};

  auto box =
      db::create<db::AddSimpleTags<Tags::Next<TemporalId>, MortarRecorderTag,
                                   mortar_next_temporal_ids_tag>>(
          self_step_end, std::move(initial_recorders),
          std::move(initial_mortar_temporal_ids));

  const Element<2> left_element(left_id,
                                {{Direction<2>::upper_xi(), {{self_id}, {}}}});
  const Element<2> right_element(right_id,
                                 {{Direction<2>::lower_xi(), {{self_id}, {}}}});

  std::vector<int> relevant_left_steps{left_steps.front()};
  for (size_t step = 1; step < left_steps.size(); ++step) {
    CHECK_FALSE(
        runner.is_ready<component, receive_data_for_fluxes>(box, self_id));
    send_from_neighbor(&runner, left_element, left_steps[step - 1],
                       left_steps[step], step);
    if (left_steps[step - 1] < self_step_end) {
      relevant_left_steps.push_back(left_steps[step]);
    }
  }
  std::vector<int> relevant_right_steps{right_steps.front()};
  for (size_t step = 1; step < right_steps.size(); ++step) {
    CHECK_FALSE(
        runner.is_ready<component, receive_data_for_fluxes>(box, self_id));
    send_from_neighbor(&runner, right_element, right_steps[step - 1],
                       right_steps[step], step);
    if (right_steps[step - 1] < self_step_end) {
      relevant_right_steps.push_back(right_steps[step]);
    }
  }
  CHECK(runner.is_ready<component, receive_data_for_fluxes>(box, self_id));

  box = std::get<0>(
      runner.apply<component, receive_data_for_fluxes>(box, self_id));

  CHECK(db::get<mortar_next_temporal_ids_tag>(box) ==
        db::item_type<mortar_next_temporal_ids_tag>{
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
