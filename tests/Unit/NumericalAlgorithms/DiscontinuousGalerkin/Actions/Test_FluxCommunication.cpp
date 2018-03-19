// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
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
  using slice_tags = tmpl::list<Tags::NormalDotFlux<Var>, OtherData>;
  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& var_flux,
                    const Scalar<DataVector>& var_flux2,
                    const Scalar<DataVector>& other_data,
                    const tnsr::i<DataVector, 2, Frame::Inertial>&
                        interface_unit_normal) const noexcept {
    CHECK(var_flux == var_flux2);
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
using send_data_for_fluxes = dg::Actions::SendDataForFluxes;
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
using mortar_data_tag = typename flux_comm_types::mortar_data_tag;
using LocalData = typename flux_comm_types::LocalData;
using PackagedData = typename flux_comm_types::PackagedData;
using MagnitudeOfFaceNormal = typename flux_comm_types::MagnitudeOfFaceNormal;
using normal_dot_fluxes_tag =
    interface_tag<typename flux_comm_types::normal_dot_fluxes_tag>;

using fluxes_tag = typename flux_comm_types::FluxesTag;

using other_data_tag = interface_tag<Tags::Variables<tmpl::list<OtherData>>>;

using compute_items = db::AddComputeTags<
    Tags::InternalDirections<2>, interface_tag<Tags::Direction<2>>,
    interface_tag<Tags::Extents<1>>,
    interface_tag<Tags::UnnormalizedFaceNormal<2>>,
    interface_tag<Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>,
    interface_tag<Tags::Normalized<
        Tags::UnnormalizedFaceNormal<2>,
        Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>>>;

Scalar<DataVector> reverse(Scalar<DataVector> x) noexcept {
  std::reverse(get(x).begin(), get(x).end());
  return x;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{NumericalFlux{}}};

  const Index<2> extents{{{3, 3}}};

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
    &extents, &self_id, &west_id, &east_id, &south_id, &block_orientation,
    &coordmap, &neighbor_directions, &data
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
    mortar_history[std::make_pair(Direction<2>::lower_xi(), west_id)];
    mortar_history[std::make_pair(Direction<2>::upper_xi(), east_id)];
    mortar_history[std::make_pair(Direction<2>::upper_eta(), south_id)];

    return db::create<
        db::AddSimpleTags<TemporalId, Tags::Extents<2>, Tags::Element<2>,
                          Tags::ElementMap<2>, normal_dot_fluxes_tag,
                          other_data_tag, mortar_data_tag>,
        compute_items>(0, extents, element, std::move(map),
                       std::move(normal_dot_fluxes), std::move(other_data),
                       std::move(mortar_history));
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
  const auto send_data = [&extents, &runner, &self_id, &coordmap](
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
        db::AddSimpleTags<TemporalId, Tags::Extents<2>, Tags::Element<2>,
                          Tags::ElementMap<2>, normal_dot_fluxes_tag,
                          other_data_tag, mortar_data_tag>,
        compute_items>(0, extents, element, std::move(map),
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
      NumericalFlux{}.package_data(&local_packaged, local_flux, local_flux,
                                   local_other, normalized_local_normal);
      local_data.assign_subset(local_packaged);

      const auto magnitude_remote_normal = magnitude(remote_normal);
      auto normalized_remote_normal = remote_normal;
      for (auto& x : normalized_remote_normal) {
        x /= get(magnitude_remote_normal);
      }
      PackagedData remote_packaged(3);
      NumericalFlux{}.package_data(&remote_packaged, remote_flux, remote_flux,
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

  const Index<2> extents{{{3, 3}}};

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  const Element<2> element(self_id, {});

  auto map = ElementMap<2, Frame::Inertial>(
      self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                   CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>(
                       {-1., 1., 3., 7.}, {-1., 1., -2., 4.})));

  auto start_box = db::create<
      db::AddSimpleTags<TemporalId, Tags::Extents<2>, Tags::Element<2>,
                        Tags::ElementMap<2>, normal_dot_fluxes_tag,
                        other_data_tag, mortar_data_tag>,
      compute_items>(0, extents, element, std::move(map),
                     db::item_type<normal_dot_fluxes_tag>{},
                     db::item_type<other_data_tag>{},
                     db::item_type<mortar_data_tag>{});

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
