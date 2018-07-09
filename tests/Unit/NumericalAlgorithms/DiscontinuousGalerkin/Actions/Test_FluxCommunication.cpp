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
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"  // for Variables

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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

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

template <size_t Dim>
struct Metavariables;
template <size_t Dim>
using send_data_for_fluxes = dg::Actions::SendDataForFluxes<Metavariables<Dim>>;
template <size_t Dim>
using receive_data_for_fluxes =
    dg::Actions::ReceiveDataForFluxes<Metavariables<Dim>>;

template <size_t Dim>
using component =
    ActionTesting::MockArrayComponent<Metavariables<Dim>, ElementIndex<Dim>,
                                      tmpl::list<NumericalFluxTag<Dim>>,
                                      tmpl::list<receive_data_for_fluxes<Dim>>>;

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<component<Dim>>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag<Dim>;
};

template <size_t Dim, typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;

template <size_t Dim>
using flux_comm_types = dg::FluxCommunicationTypes<Metavariables<Dim>>;
template <size_t Dim>
using mortar_data_tag = typename flux_comm_types<Dim>::simple_mortar_data_tag;
template <size_t Dim>
using LocalData = typename flux_comm_types<Dim>::LocalData;
template <size_t Dim>
using LocalMortarData = typename flux_comm_types<Dim>::LocalMortarData;
template <size_t Dim>
using PackagedData = typename flux_comm_types<Dim>::PackagedData;
template <size_t Dim>
using normal_dot_fluxes_tag =
    interface_tag<Dim, typename flux_comm_types<Dim>::normal_dot_fluxes_tag>;

template <size_t Dim>
using fluxes_tag = typename flux_comm_types<Dim>::FluxesTag;

template <size_t Dim>
using other_data_tag =
    interface_tag<Dim, Tags::Variables<tmpl::list<OtherData>>>;
template <size_t Dim>
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, Dim>;
template <size_t Dim>
using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>;
template <size_t Dim>
using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>;

template <size_t Dim>
using compute_items = db::AddComputeTags<
    Tags::InternalDirections<Dim>, interface_tag<Dim, Tags::Direction<Dim>>,
    interface_tag<Dim, Tags::Mesh<Dim - 1>>,
    interface_tag<Dim, Tags::UnnormalizedFaceNormal<Dim>>,
    interface_tag<Dim,
                  Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
    interface_tag<Dim, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

Scalar<DataVector> reverse(Scalar<DataVector> x) noexcept {
  std::reverse(get(x).begin(), get(x).end());
  return x;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables<2>> runner{{NumericalFlux<2>{}}};

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

    db::item_type<normal_dot_fluxes_tag<2>> normal_dot_fluxes;
    for (const auto& direction : neighbor_directions) {
      auto& flux_vars = normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<Var>>(flux_vars) = data.fluxes.at(direction);
    }

    db::item_type<other_data_tag<2>> other_data;
    for (const auto& direction : neighbor_directions) {
      auto& other_data_vars = other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) = data.other_data.at(direction);
    }

    db::item_type<mortar_data_tag<2>> mortar_history{};
    db::item_type<mortar_next_temporal_ids_tag<2>> mortar_next_temporal_ids{};
    db::item_type<mortar_meshes_tag<2>> mortar_meshes{};
    db::item_type<mortar_sizes_tag<2>> mortar_sizes{};
    for (const auto& mortar_id : neighbor_mortar_ids) {
      mortar_history.insert({mortar_id, {}});
      mortar_next_temporal_ids.insert({mortar_id, 0});
      mortar_meshes.insert({mortar_id, mesh.slice_away(0)});
      mortar_sizes.insert({mortar_id, {{Spectral::MortarSize::Full}}});
    }

    return db::create<
        db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>,
                          Tags::Element<2>, Tags::ElementMap<2>,
                          normal_dot_fluxes_tag<2>, other_data_tag<2>,
                          mortar_data_tag<2>, mortar_next_temporal_ids_tag<2>,
                          mortar_meshes_tag<2>, mortar_sizes_tag<2>>,
        compute_items<2>>(0, 1, mesh, element, std::move(map),
                          std::move(normal_dot_fluxes), std::move(other_data),
                          std::move(mortar_history),
                          std::move(mortar_next_temporal_ids),
                          std::move(mortar_meshes), std::move(mortar_sizes));
  }();

  auto sent_box = std::get<0>(
      runner.apply<component<2>, send_data_for_fluxes<2>>(start_box, self_id));

  // Here, we just check that messages are sent to the correct places.
  // We will check the received values on the central element later.
  {
    CHECK(runner.nonempty_inboxes<component<2>, fluxes_tag<2>>() ==
          std::unordered_set<ElementIndex<2>>{west_id, east_id, south_id});
    const auto check_sent_data = [&runner, &self_id](
        const ElementId<2>& id, const Direction<2>& direction) noexcept {
      const auto& inboxes = runner.inboxes<component<2>>();
      const auto& flux_inbox = tuples::get<fluxes_tag<2>>(inboxes.at(id));
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

    db::item_type<normal_dot_fluxes_tag<2>> normal_dot_fluxes_map{};
    normal_dot_fluxes_map[direction].initialize(get(normal_dot_fluxes).size());
    get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes_map[direction]) =
        normal_dot_fluxes;

    db::item_type<other_data_tag<2>> other_data_map{};
    other_data_map[direction].initialize(get(other_data).size());
    get<OtherData>(other_data_map[direction]) = other_data;

    db::item_type<mortar_data_tag<2>> mortar_history{};
    mortar_history[std::make_pair(direction, self_id)];

    db::item_type<mortar_meshes_tag<2>> mortar_meshes{};
    mortar_meshes.insert({{direction, self_id}, mesh.slice_away(0)});

    db::item_type<mortar_sizes_tag<2>> mortar_sizes{};
    mortar_sizes.insert({{direction, self_id}, {{Spectral::MortarSize::Full}}});

    auto box = db::create<
        db::AddSimpleTags<
            TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>, Tags::Element<2>,
            Tags::ElementMap<2>, normal_dot_fluxes_tag<2>, other_data_tag<2>,
            mortar_data_tag<2>, mortar_meshes_tag<2>, mortar_sizes_tag<2>>,
        compute_items<2>>(0, 1, mesh, element, std::move(map),
                          std::move(normal_dot_fluxes_map),
                          std::move(other_data_map), std::move(mortar_history),
                          std::move(mortar_meshes), std::move(mortar_sizes));

    runner.apply<component<2>, send_data_for_fluxes<2>>(box, id);
  };

  CHECK_FALSE(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(
      sent_box, self_id));
  send_data(south_id, Direction<2>::lower_eta(), {},
            data.remote_fluxes.at(Direction<2>::upper_eta()),
            data.remote_other_data.at(Direction<2>::upper_eta()));
  CHECK_FALSE(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(
      sent_box, self_id));
  send_data(east_id, Direction<2>::lower_xi(), {},
            data.remote_fluxes.at(Direction<2>::upper_xi()),
            data.remote_other_data.at(Direction<2>::upper_xi()));
  CHECK_FALSE(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(
      sent_box, self_id));
  send_data(west_id, Direction<2>::lower_eta(), block_orientation.inverse_map(),
            data.remote_fluxes.at(Direction<2>::lower_xi()),
            data.remote_other_data.at(Direction<2>::lower_xi()));
  CHECK(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(sent_box,
                                                                  self_id));

  auto received_box =
      std::get<0>(runner.apply<component<2>, receive_data_for_fluxes<2>>(
          sent_box, self_id));

  CHECK(tuples::get<fluxes_tag<2>>(runner.inboxes<component<2>>()[self_id])
            .empty());

  for (const auto& mortar_id : neighbor_mortar_ids) {
    CHECK(
        db::get<mortar_next_temporal_ids_tag<2>>(received_box).at(mortar_id) ==
        1);
  };

  auto mortar_history =
      serialize_and_deserialize(db::get<mortar_data_tag<2>>(received_box));
  CHECK(mortar_history.size() == 3);
  const auto check_mortar = [&mortar_history](
      const std::pair<Direction<2>, ElementId<2>>& mortar_id,
      const Scalar<DataVector>& local_flux,
      const Scalar<DataVector>& remote_flux,
      const Scalar<DataVector>& local_other,
      const Scalar<DataVector>& remote_other,
      const tnsr::i<DataVector, 2>& local_normal,
      const tnsr::i<DataVector, 2>& remote_normal) noexcept {
    LocalMortarData<2> local_mortar_data(3);
    get<Tags::NormalDotFlux<Var>>(local_mortar_data) = local_flux;
    const auto magnitude_local_normal = magnitude(local_normal);
    auto normalized_local_normal = local_normal;
    for (auto& x : normalized_local_normal) {
      x /= get(magnitude_local_normal);
    }
    PackagedData<2> local_packaged(3);
    NumericalFlux<2>{}.package_data(&local_packaged, local_flux, local_other,
                                    normalized_local_normal);
    local_mortar_data.assign_subset(local_packaged);

    const auto magnitude_remote_normal = magnitude(remote_normal);
    auto normalized_remote_normal = remote_normal;
    for (auto& x : normalized_remote_normal) {
      x /= get(magnitude_remote_normal);
    }
    PackagedData<2> remote_packaged(3);
    NumericalFlux<2>{}.package_data(&remote_packaged, remote_flux, remote_other,
                                    normalized_remote_normal);

    const auto result = mortar_history.at(mortar_id).extract();
    CHECK(result.first.mortar_data == local_mortar_data);
    CHECK(result.first.magnitude_of_face_normal == magnitude_local_normal);
    CHECK(result.second == remote_packaged);
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
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.NoNeighbors",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables<2>> runner{{NumericalFlux<2>{}}};

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
                        normal_dot_fluxes_tag<2>, other_data_tag<2>,
                        mortar_data_tag<2>, mortar_next_temporal_ids_tag<2>,
                        mortar_meshes_tag<2>, mortar_sizes_tag<2>>,
      compute_items<2>>(0, 1, mesh, element, std::move(map),
                        db::item_type<normal_dot_fluxes_tag<2>>{},
                        db::item_type<other_data_tag<2>>{},
                        db::item_type<mortar_data_tag<2>>{},
                        db::item_type<mortar_next_temporal_ids_tag<2>>{},
                        db::item_type<mortar_meshes_tag<2>>{},
                        db::item_type<mortar_sizes_tag<2>>{});

  auto sent_box = std::get<0>(
      runner.apply<component<2>, send_data_for_fluxes<2>>(start_box, self_id));

  CHECK(db::get<mortar_data_tag<2>>(sent_box).empty());
  CHECK(runner.nonempty_inboxes<component<2>, fluxes_tag<2>>().empty());

  CHECK(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(sent_box,
                                                                  self_id));

  const auto received_box =
      std::get<0>(runner.apply<component<2>, receive_data_for_fluxes<2>>(
          sent_box, self_id));

  CHECK(db::get<mortar_data_tag<2>>(received_box).empty());
  CHECK(runner.nonempty_inboxes<component<2>, fluxes_tag<2>>().empty());
}

namespace {
struct DataRecorder {
  // Only called on the sending sides, which are not interesting here.
  void local_insert(int /*temporal_id*/,
                    const LocalData<2>& /*data*/) noexcept {}

  void remote_insert(int temporal_id, PackagedData<2> data) noexcept {
    received_data.emplace_back(temporal_id, std::move(data));
  }

  std::vector<std::pair<int, PackagedData<2>>> received_data{};
};

struct DataRecorderTag : db::SimpleTag {
  static std::string name() { return "DataRecorderTag"; }
  using type = DataRecorder;
};

struct MortarRecorderTag : Tags::VariablesBoundaryData,
                           Tags::Mortars<DataRecorderTag, 2> {};

void send_from_neighbor(
    const gsl::not_null<ActionTesting::ActionRunner<Metavariables<2>>*> runner,
    const Element<2>& element, const int start, const int end,
    const double n_dot_f) noexcept {
  const Direction<2>& send_direction = element.neighbors().begin()->first;
  const ElementId<2>& receiver_id =
      *element.neighbors().begin()->second.begin();

  const Mesh<2> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  ElementMap<2, Frame::Inertial> map(
      element.id(), make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                        CoordinateMaps::Identity<2>{}));

  db::item_type<normal_dot_fluxes_tag<2>> fluxes;
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
          Tags::ElementMap<2>, normal_dot_fluxes_tag<2>, other_data_tag<2>,
          mortar_meshes_tag<2>, mortar_sizes_tag<2>, MortarRecorderTag>,
      compute_items<2>>(start, end, mesh, element, std::move(map),
                        std::move(fluxes), std::move(other_data),
                        std::move(mortar_meshes), std::move(mortar_sizes),
                        std::move(recorders));

  runner->apply<component<2>, send_data_for_fluxes<2>>(box, element.id());
}

// Sends the left steps, then the right steps.  The step must not be
// ready until after the last send.
void run_lts_case(const int self_step_end, const std::vector<int>& left_steps,
                  const std::vector<int>& right_steps) noexcept {
  ActionTesting::ActionRunner<Metavariables<2>> runner{{NumericalFlux<2>{}}};

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

  auto box =
      db::create<db::AddSimpleTags<Tags::Next<TemporalId>, MortarRecorderTag,
                                   mortar_next_temporal_ids_tag<2>>>(
          self_step_end, std::move(initial_recorders),
          std::move(initial_mortar_temporal_ids));

  const Element<2> left_element(left_id,
                                {{Direction<2>::upper_xi(), {{self_id}, {}}}});
  const Element<2> right_element(right_id,
                                 {{Direction<2>::lower_xi(), {{self_id}, {}}}});

  std::vector<int> relevant_left_steps{left_steps.front()};
  for (size_t step = 1; step < left_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(
        box, self_id));
    send_from_neighbor(&runner, left_element, left_steps[step - 1],
                       left_steps[step], step);
    if (left_steps[step - 1] < self_step_end) {
      relevant_left_steps.push_back(left_steps[step]);
    }
  }
  std::vector<int> relevant_right_steps{right_steps.front()};
  for (size_t step = 1; step < right_steps.size(); ++step) {
    CHECK_FALSE(runner.is_ready<component<2>, receive_data_for_fluxes<2>>(
        box, self_id));
    send_from_neighbor(&runner, right_element, right_steps[step - 1],
                       right_steps[step], step);
    if (right_steps[step - 1] < self_step_end) {
      relevant_right_steps.push_back(right_steps[step]);
    }
  }
  CHECK(
      runner.is_ready<component<2>, receive_data_for_fluxes<2>>(box, self_id));

  box = std::get<0>(
      runner.apply<component<2>, receive_data_for_fluxes<2>>(box, self_id));

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

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.p-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables<3>> runner{{NumericalFlux<3>{}}};

  const ElementId<3> self_id(1);
  const ElementId<3> neighbor_id(2);

  const auto mortar_id = std::make_pair(Direction<3>::upper_eta(), neighbor_id);
  const Element<3> element(
      self_id, {{mortar_id.first,
                 {{neighbor_id},
                  OrientationMap<3>{
                      {{Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
                        Direction<3>::lower_eta()}}}}}});

  ElementMap<3, Frame::Inertial> map(
      self_id,
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Identity<3>{}));

  const Mesh<3> mesh({{2, 3, 4}}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const Mesh<2> face_mesh = mesh.slice_away(mortar_id.first.dimension());
  const Mesh<2> mortar_mesh({{5, 6}}, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const Mesh<2> rotated_mortar_mesh({{6, 5}}, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto);

  const auto face_coords = logical_coordinates(face_mesh);
  const auto mortar_coords = logical_coordinates(mortar_mesh);
  const auto rotated_mortar_coords = logical_coordinates(rotated_mortar_mesh);

  const auto flux = [](const DataVector& x, const DataVector& y) noexcept {
    return Scalar<DataVector>(x * cube(y));
  };
  const auto packaged_data = [](const Scalar<DataVector>& var_flux) noexcept {
    PackagedData<3> packaged(get(var_flux).size());
    const tnsr::i<DataVector, 3, Frame::Inertial> normal(get(var_flux).size(),
                                                         1.);
    NumericalFlux<3>{}.package_data(&packaged, var_flux, var_flux, normal);
    return packaged;
  };

  db::item_type<normal_dot_fluxes_tag<3>> normal_dot_fluxes;
  normal_dot_fluxes[mortar_id.first].initialize(
      face_mesh.number_of_grid_points());
  get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes[mortar_id.first]) =
      flux(get<0>(face_coords), get<1>(face_coords));

  // Value does not affect tested results
  db::item_type<other_data_tag<3>> other_data;
  other_data[mortar_id.first].initialize(face_mesh.number_of_grid_points(), 0.);

  auto start_box = db::create<
      db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<3>,
                        Tags::Element<3>, Tags::ElementMap<3>,
                        normal_dot_fluxes_tag<3>, other_data_tag<3>,
                        mortar_data_tag<3>, mortar_next_temporal_ids_tag<3>,
                        mortar_meshes_tag<3>, mortar_sizes_tag<3>>,
      compute_items<3>>(
      0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes),
      std::move(other_data), db::item_type<mortar_data_tag<3>>{{mortar_id, {}}},
      db::item_type<mortar_next_temporal_ids_tag<3>>{{mortar_id, 1}},
      db::item_type<mortar_meshes_tag<3>>{{mortar_id, mortar_mesh}},
      db::item_type<mortar_sizes_tag<3>>{
          {mortar_id,
           {{Spectral::MortarSize::Full, Spectral::MortarSize::Full}}}});

  auto sent_box = std::get<0>(
      runner.apply<component<3>, send_data_for_fluxes<3>>(start_box, self_id));

  // Check local data
  {
    CHECK(db::get<mortar_data_tag<3>>(sent_box).size() == 1);
    auto mortar_data = db::get<mortar_data_tag<3>>(sent_box).at(mortar_id);
    mortar_data.remote_insert(0, PackagedData<3>{});
    const auto local_data = mortar_data.extract().first;
    CHECK(local_data.mortar_data.number_of_grid_points() ==
          mortar_mesh.number_of_grid_points());
    CHECK(get(local_data.magnitude_of_face_normal).size() ==
          face_mesh.number_of_grid_points());

    CHECK_ITERABLE_APPROX(get<Var>(local_data.mortar_data),
                          get<Var>(packaged_data(flux(get<0>(mortar_coords),
                                                      get<1>(mortar_coords)))));

    CHECK_ITERABLE_APPROX(get<Tags::NormalDotFlux<Var>>(local_data.mortar_data),
                          flux(get<0>(mortar_coords), get<1>(mortar_coords)));
    CHECK(get(local_data.magnitude_of_face_normal) ==
          DataVector(face_mesh.number_of_grid_points(), 1.));
  }

  // Check sent data
  {
    CHECK(runner.nonempty_inboxes<component<3>, fluxes_tag<3>>().size() == 1);
    const auto& inbox = tuples::get<fluxes_tag<3>>(
        runner.inboxes<component<3>>().at(neighbor_id));

    const auto& received_flux =
        inbox.at(0).at({Direction<3>::upper_xi(), self_id}).second;
    CHECK_ITERABLE_APPROX(
        get<Var>(received_flux),
        get<Var>(packaged_data(flux(get<1>(rotated_mortar_coords),
                                    -get<0>(rotated_mortar_coords)))));
  }
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.h-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables<2>> runner{{NumericalFlux<2>{}}};

  const Scalar<DataVector> n_dot_f{{{{2., 3.}}}};
  for (const auto& test :
       {std::make_pair(Spectral::MortarSize::Full, DataVector{2., 3.}),
        std::make_pair(Spectral::MortarSize::LowerHalf, DataVector{2., 2.5}),
        std::make_pair(Spectral::MortarSize::UpperHalf, DataVector{2.5, 3.})}) {
    CAPTURE(test.first);

    const ElementId<2> self_id(1);
    const ElementId<2> neighbor_id(2);

    const auto mortar_id =
        std::make_pair(Direction<2>::upper_xi(), neighbor_id);
    const Element<2> element(
        self_id, {{mortar_id.first,
                   {{neighbor_id},
                    OrientationMap<2>{{{Direction<2>::upper_xi(),
                                        Direction<2>::lower_eta()}}}}}});

    ElementMap<2, Frame::Inertial> map(
        self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                    CoordinateMaps::Affine>(
                         {-1., 1., -1., 1.}, {-1., 1., -1., 1.})));

    const Mesh<2> mesh(2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    const Mesh<1> face_mesh = mesh.slice_away(mortar_id.first.dimension());

    const auto packaged_data = [](const DataVector& var_flux) noexcept {
      const Scalar<DataVector> scalar_flux(var_flux);
      PackagedData<2> packaged(var_flux.size());
      const tnsr::i<DataVector, 2, Frame::Inertial> normal(var_flux.size(), 1.);
      NumericalFlux<2>{}.package_data(&packaged, scalar_flux, scalar_flux,
                                      normal);
      return packaged;
    };

    db::item_type<normal_dot_fluxes_tag<2>> normal_dot_fluxes;
    normal_dot_fluxes[mortar_id.first].initialize(
        face_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes[mortar_id.first]) = n_dot_f;

    // Value does not affect tested results
    db::item_type<other_data_tag<2>> other_data;
    other_data[mortar_id.first].initialize(face_mesh.number_of_grid_points(),
                                           0.);

    auto start_box = db::create<
        db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>,
                          Tags::Element<2>, Tags::ElementMap<2>,
                          normal_dot_fluxes_tag<2>, other_data_tag<2>,
                          mortar_data_tag<2>, mortar_next_temporal_ids_tag<2>,
                          mortar_meshes_tag<2>, mortar_sizes_tag<2>>,
        compute_items<2>>(
        0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes),
        std::move(other_data),
        db::item_type<mortar_data_tag<2>>{{mortar_id, {}}},
        db::item_type<mortar_next_temporal_ids_tag<2>>{{mortar_id, 1}},
        db::item_type<mortar_meshes_tag<2>>{{mortar_id, face_mesh}},
        db::item_type<mortar_sizes_tag<2>>{{mortar_id, {{test.first}}}});

    auto sent_box =
        std::get<0>(runner.apply<component<2>, send_data_for_fluxes<2>>(
            start_box, self_id));

    // Check local data
    {
      CHECK(db::get<mortar_data_tag<2>>(sent_box).size() == 1);
      auto mortar_data = db::get<mortar_data_tag<2>>(sent_box).at(mortar_id);
      mortar_data.remote_insert(0, PackagedData<2>{});
      const auto local_data = mortar_data.extract().first;
      CHECK(local_data.mortar_data.number_of_grid_points() ==
            face_mesh.number_of_grid_points());
      CHECK(get(local_data.magnitude_of_face_normal).size() ==
            face_mesh.number_of_grid_points());

      CHECK_ITERABLE_APPROX(get<Var>(local_data.mortar_data),
                            get<Var>(packaged_data(test.second)));

      CHECK(get<Tags::NormalDotFlux<Var>>(local_data.mortar_data) == n_dot_f);
      CHECK(get(local_data.magnitude_of_face_normal) ==
            DataVector(face_mesh.number_of_grid_points(), 1.));
    }

    // Check sent data
    {
      CHECK(runner.nonempty_inboxes<component<2>, fluxes_tag<2>>().size() == 1);
      auto& inbox = tuples::get<fluxes_tag<2>>(
          runner.inboxes<component<2>>().at(neighbor_id));

      const auto& received_flux =
          inbox.at(0).at({Direction<2>::lower_xi(), self_id}).second;
      // The interface has an inverting orientation.
      CHECK_ITERABLE_APPROX(
          get<Var>(received_flux),
          get<Var>(packaged_data({test.second[1], test.second[0]})));
      inbox.clear();
    }
  }
}
