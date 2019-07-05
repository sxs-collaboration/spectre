// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>  // IWYU pragma: keep
#include <map>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>
// IWYU pragma: no_include <boost/variant/get.hpp>

// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"  // for Variables
// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
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
struct NumericalFluxTag : db::SimpleTag {
  using type = NumericalFlux<Dim>;
  using container_tag = NumericalFluxTag;
  static std::string name() noexcept { return "NumericalFluxTag"; }
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
using mortar_data_tag = typename FluxCommTypes::simple_mortar_data_tag;
template <typename FluxCommTypes>
using LocalMortarData = typename FluxCommTypes::LocalMortarData;
template <typename FluxCommTypes>
using PackagedData = typename FluxCommTypes::PackagedData;
template <size_t Dim, typename FluxCommTypes>
using normal_dot_fluxes_tag =
    interface_tag<Dim, typename FluxCommTypes::normal_dot_fluxes_tag>;

template <typename FluxCommTypes>
using fluxes_tag = typename FluxCommTypes::FluxesTag;

template <size_t Dim>
using other_data_tag =
    interface_tag<Dim, Tags::Variables<tmpl::list<OtherData>>>;
template <size_t Dim>
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, Dim>;
template <size_t Dim>
using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>;
template <size_t Dim>
using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>;

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<NumericalFluxTag<Dim>>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;

  using simple_tags =
      db::AddSimpleTags<TemporalId, Tags::Next<TemporalId>, Tags::Mesh<Dim>,
                        Tags::Element<Dim>, Tags::ElementMap<Dim>,
                        normal_dot_fluxes_tag<Dim, flux_comm_types>,
                        other_data_tag<Dim>, mortar_data_tag<flux_comm_types>,
                        mortar_next_temporal_ids_tag<Dim>,
                        mortar_meshes_tag<Dim>, mortar_sizes_tag<Dim>>;

  using compute_tags = db::AddComputeTags<
      Tags::InternalDirections<Dim>,
      interface_compute_tag<Dim, Tags::Direction<Dim>>,
      interface_compute_tag<Dim, Tags::InterfaceMesh<Dim>>,
      interface_compute_tag<Dim, Tags::UnnormalizedFaceNormal<Dim>>,
      interface_compute_tag<
          Dim, Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
      interface_compute_tag<
          Dim, Tags::NormalizedCompute<Tags::UnnormalizedFaceNormal<Dim>>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<dg::Actions::SendDataForFluxes<Metavariables>,
                     dg::Actions::ReceiveDataForFluxes<Metavariables>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag<Dim>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Component>
using compute_items = typename Component::compute_tags;

template <size_t Dim>
using flux_comm_types =
    typename component<Dim, Metavariables<Dim>>::flux_comm_types;

Scalar<DataVector> reverse(Scalar<DataVector> x) noexcept {
  std::reverse(get(x).begin(), get(x).end());
  return x;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;
  using simple_tags = typename my_component::simple_tags;
  using compute_tags = typename my_component::compute_tags;
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
  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));

  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(xi_map, eta_map));

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

  ActionTesting::MockRuntimeSystem<metavariables> runner{{NumericalFlux<2>{}}};

  // Emplace self element
  {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>
        normal_dot_fluxes;
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

    db::item_type<mortar_data_tag<flux_comm_types<2>>> mortar_history{};
    db::item_type<mortar_next_temporal_ids_tag<2>> mortar_next_temporal_ids{};
    db::item_type<mortar_meshes_tag<2>> mortar_meshes{};
    db::item_type<mortar_sizes_tag<2>> mortar_sizes{};
    for (const auto& mortar_id : neighbor_mortar_ids) {
      mortar_history.insert({mortar_id, {}});
      mortar_next_temporal_ids.insert({mortar_id, 0});
      mortar_meshes.insert({mortar_id, mesh.slice_away(0)});
      mortar_sizes.insert({mortar_id, {{Spectral::MortarSize::Full}}});
    }

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, self_id,
        {0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes),
         std::move(other_data), std::move(mortar_history),
         std::move(mortar_next_temporal_ids), std::move(mortar_meshes),
         std::move(mortar_sizes)});
  }

  const auto emplace_neighbor = [&mesh, &self_id, &coordmap, &runner ](
      const ElementId<2>& id, const Direction<2>& direction,
      const OrientationMap<2>& orientation,
      const Scalar<DataVector>& normal_dot_fluxes,
      const Scalar<DataVector>& other_data) noexcept {
    const Element<2> element(id, {{direction, {{self_id}, orientation}}});
    auto map = ElementMap<2, Frame::Inertial>(id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>
        normal_dot_fluxes_map{};
    normal_dot_fluxes_map[direction].initialize(get(normal_dot_fluxes).size());
    get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes_map[direction]) =
        normal_dot_fluxes;

    db::item_type<other_data_tag<2>> other_data_map{};
    other_data_map[direction].initialize(get(other_data).size());
    get<OtherData>(other_data_map[direction]) = other_data;

    db::item_type<mortar_data_tag<flux_comm_types<2>>> mortar_history{};
    mortar_history[std::make_pair(direction, self_id)];

    db::item_type<mortar_meshes_tag<2>> mortar_meshes{};
    mortar_meshes.insert({{direction, self_id}, mesh.slice_away(0)});

    db::item_type<mortar_sizes_tag<2>> mortar_sizes{};
    mortar_sizes.insert({{direction, self_id}, {{Spectral::MortarSize::Full}}});

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, id,
        {0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes_map),
         std::move(other_data_map), std::move(mortar_history),
         db::item_type<mortar_next_temporal_ids_tag<2>>{},
         std::move(mortar_meshes), std::move(mortar_sizes)});
  };

  emplace_neighbor(south_id, Direction<2>::lower_eta(), {},
                   data.remote_fluxes.at(Direction<2>::upper_eta()),
                   data.remote_other_data.at(Direction<2>::upper_eta()));
  emplace_neighbor(east_id, Direction<2>::lower_xi(), {},
                   data.remote_fluxes.at(Direction<2>::upper_xi()),
                   data.remote_other_data.at(Direction<2>::upper_xi()));
  emplace_neighbor(west_id, Direction<2>::lower_eta(),
                   block_orientation.inverse_map(),
                   data.remote_fluxes.at(Direction<2>::lower_xi()),
                   data.remote_other_data.at(Direction<2>::lower_xi()));
  runner.set_phase(metavariables::Phase::Testing);

  runner.next_action<my_component>(self_id);

  // Here, we just check that messages are sent to the correct places.
  // We will check the received values on the central element later.
  {
    CHECK(
        runner
            .nonempty_inboxes<my_component, fluxes_tag<flux_comm_types<2>>>() ==
        std::unordered_set<ElementIndex<2>>{west_id, east_id, south_id});
    const auto check_sent_data = [&runner, &self_id ](
        const ElementId<2>& id, const Direction<2>& direction) noexcept {
      const auto& inboxes = runner.inboxes<my_component>();
      const auto& flux_inbox =
          tuples::get<fluxes_tag<flux_comm_types<2>>>(inboxes.at(id));
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
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(south_id);
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(east_id);
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(west_id);
  CHECK(runner.is_ready<my_component>(self_id));

  // ReceiveDataForFluxes
  runner.next_action<my_component>(self_id);
  const auto& self_box =
      ActionTesting::get_databox<my_component,
                                 tmpl::append<simple_tags, compute_tags>>(
          make_not_null(&runner), self_id);

  CHECK(tuples::get<fluxes_tag<flux_comm_types<2>>>(
            runner.inboxes<my_component>()[self_id])
            .empty());

  for (const auto& mortar_id : neighbor_mortar_ids) {
    CHECK(db::get<mortar_next_temporal_ids_tag<2>>(self_box).at(mortar_id) ==
          1);
  };

  auto mortar_history = serialize_and_deserialize(
      db::get<mortar_data_tag<flux_comm_types<2>>>(self_box));
  CHECK(mortar_history.size() == 3);
  const auto check_mortar = [&mortar_history](
      const std::pair<Direction<2>, ElementId<2>>& mortar_id,
      const Scalar<DataVector>& local_flux,
      const Scalar<DataVector>& remote_flux,
      const Scalar<DataVector>& local_other,
      const Scalar<DataVector>& remote_other,
      const tnsr::i<DataVector, 2>& local_normal,
      const tnsr::i<DataVector, 2>& remote_normal) noexcept {
    LocalMortarData<flux_comm_types<2>> local_mortar_data(3);
    get<Tags::NormalDotFlux<Var>>(local_mortar_data) = local_flux;
    const auto magnitude_local_normal = magnitude(local_normal);
    auto normalized_local_normal = local_normal;
    for (auto& x : normalized_local_normal) {
      x /= get(magnitude_local_normal);
    }
    PackagedData<flux_comm_types<2>> local_packaged(3);
    NumericalFlux<2>{}.package_data(&local_packaged, local_flux, local_other,
                                    normalized_local_normal);
    local_mortar_data.assign_subset(local_packaged);

    const auto magnitude_remote_normal = magnitude(remote_normal);
    auto normalized_remote_normal = remote_normal;
    for (auto& x : normalized_remote_normal) {
      x /= get(magnitude_remote_normal);
    }
    PackagedData<flux_comm_types<2>> remote_packaged(3);
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
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
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

  ActionTesting::MockRuntimeSystem<metavariables> runner{{NumericalFlux<2>{}}};
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {0, 1, mesh, element, std::move(map),
       db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>{},
       db::item_type<other_data_tag<2>>{},
       db::item_type<mortar_data_tag<flux_comm_types<2>>>{},
       db::item_type<mortar_next_temporal_ids_tag<2>>{},
       db::item_type<mortar_meshes_tag<2>>{},
       db::item_type<mortar_sizes_tag<2>>{}});
  runner.set_phase(metavariables::Phase::Testing);

  runner.next_action<my_component>(self_id);

  CHECK(ActionTesting::get_databox_tag<my_component,
                                       mortar_data_tag<flux_comm_types<2>>>(
            runner, self_id)
            .empty());
  CHECK(runner.nonempty_inboxes<my_component, fluxes_tag<flux_comm_types<2>>>()
            .empty());

  CHECK(runner.is_ready<my_component>(self_id));

  runner.next_action<my_component>(self_id);

  CHECK(ActionTesting::get_databox_tag<my_component,
                                       mortar_data_tag<flux_comm_types<2>>>(
            runner, self_id)
            .empty());
  CHECK(runner.nonempty_inboxes<my_component, fluxes_tag<flux_comm_types<2>>>()
            .empty());
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.p-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<3>;
  using my_component = component<3, metavariables>;

  const ElementId<3> self_id(1);
  const ElementId<3> neighbor_id(2);

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
    PackagedData<flux_comm_types<3>> packaged(get(var_flux).size());
    const tnsr::i<DataVector, 3, Frame::Inertial> normal(get(var_flux).size(),
                                                         1.);
    NumericalFlux<3>{}.package_data(&packaged, var_flux, var_flux, normal);
    return packaged;
  };

  db::item_type<normal_dot_fluxes_tag<3, flux_comm_types<3>>> normal_dot_fluxes;
  normal_dot_fluxes[mortar_id.first].initialize(
      face_mesh.number_of_grid_points());
  get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes[mortar_id.first]) =
      flux(get<0>(face_coords), get<1>(face_coords));

  // Value does not affect tested results
  db::item_type<other_data_tag<3>> other_data;
  other_data[mortar_id.first].initialize(face_mesh.number_of_grid_points(), 0.);

  using simple_tags = typename my_component::simple_tags;
  using compute_tags = typename my_component::compute_tags;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{NumericalFlux<3>{}}};
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes),
       std::move(other_data),
       db::item_type<mortar_data_tag<flux_comm_types<3>>>{{mortar_id, {}}},
       db::item_type<mortar_next_temporal_ids_tag<3>>{{mortar_id, 1}},
       db::item_type<mortar_meshes_tag<3>>{{mortar_id, mortar_mesh}},
       db::item_type<mortar_sizes_tag<3>>{
           {mortar_id,
            {{Spectral::MortarSize::Full, Spectral::MortarSize::Full}}}}});
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, neighbor_id,
      {0, 1, mesh, element, ElementMap<3, Frame::Inertial>{},
       db::item_type<normal_dot_fluxes_tag<3, flux_comm_types<3>>>{},
       db::item_type<other_data_tag<3>>{},
       db::item_type<mortar_data_tag<flux_comm_types<3>>>{{mortar_id, {}}},
       db::item_type<mortar_next_temporal_ids_tag<3>>{{mortar_id, 1}},
       db::item_type<mortar_meshes_tag<3>>{{mortar_id, face_mesh}},
       db::item_type<mortar_sizes_tag<3>>{{mortar_id, {{}}}}});
  runner.set_phase(metavariables::Phase::Testing);

  runner.next_action<my_component>(self_id);

  // Check local data
  {
    const auto& sent_box =
        ActionTesting::get_databox<my_component,
                                   tmpl::append<simple_tags, compute_tags>>(
            runner, self_id);
    CHECK(db::get<mortar_data_tag<flux_comm_types<3>>>(sent_box).size() == 1);
    auto mortar_data =
        db::get<mortar_data_tag<flux_comm_types<3>>>(sent_box).at(mortar_id);
    mortar_data.remote_insert(0, PackagedData<flux_comm_types<3>>{});
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
    CHECK(
        runner.nonempty_inboxes<my_component, fluxes_tag<flux_comm_types<3>>>()
            .size() == 1);
    const auto& inbox = tuples::get<fluxes_tag<flux_comm_types<3>>>(
        runner.inboxes<my_component>().at(neighbor_id));

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
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;
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
    const Mesh<1> face_mesh = mesh.slice_away(mortar_id.first.dimension());

    const auto packaged_data = [](const DataVector& var_flux) noexcept {
      const Scalar<DataVector> scalar_flux(var_flux);
      PackagedData<flux_comm_types<2>> packaged(var_flux.size());
      const tnsr::i<DataVector, 2, Frame::Inertial> normal(var_flux.size(), 1.);
      NumericalFlux<2>{}.package_data(&packaged, scalar_flux, scalar_flux,
                                      normal);
      return packaged;
    };

    db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>
        normal_dot_fluxes;
    normal_dot_fluxes[mortar_id.first].initialize(
        face_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes[mortar_id.first]) = n_dot_f;

    // Value does not affect tested results
    db::item_type<other_data_tag<2>> other_data;
    other_data[mortar_id.first].initialize(face_mesh.number_of_grid_points(),
                                           0.);

    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {NumericalFlux<2>{}}};

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, self_id,
        {0, 1, mesh, element, std::move(map), std::move(normal_dot_fluxes),
         std::move(other_data),
         db::item_type<mortar_data_tag<flux_comm_types<2>>>{{mortar_id, {}}},
         db::item_type<mortar_next_temporal_ids_tag<2>>{{mortar_id, 1}},
         db::item_type<mortar_meshes_tag<2>>{{mortar_id, face_mesh}},
         db::item_type<mortar_sizes_tag<2>>{{mortar_id, {{test.first}}}}});
    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, neighbor_id,
        {0, 1, mesh, element, ElementMap<2, Frame::Inertial>{},
         db::item_type<normal_dot_fluxes_tag<2, flux_comm_types<2>>>{},
         db::item_type<other_data_tag<2>>{},
         db::item_type<mortar_data_tag<flux_comm_types<2>>>{{mortar_id, {}}},
         db::item_type<mortar_next_temporal_ids_tag<2>>{{mortar_id, 1}},
         db::item_type<mortar_meshes_tag<2>>{{mortar_id, face_mesh}},
         db::item_type<mortar_sizes_tag<2>>{{mortar_id, {{test.first}}}}});
    runner.set_phase(metavariables::Phase::Testing);

    runner.next_action<my_component>(self_id);

    // Check local data
    {
      CHECK(ActionTesting::get_databox_tag<my_component,
                                           mortar_data_tag<flux_comm_types<2>>>(
                runner, self_id)
                .size() == 1);
      auto mortar_data =
          ActionTesting::get_databox_tag<my_component,
                                         mortar_data_tag<flux_comm_types<2>>>(
              runner, self_id)
              .at(mortar_id);
      mortar_data.remote_insert(0, PackagedData<flux_comm_types<2>>{});
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
      CHECK(
          runner
              .nonempty_inboxes<my_component, fluxes_tag<flux_comm_types<2>>>()
              .size() == 1);
      auto& inbox = tuples::get<fluxes_tag<flux_comm_types<2>>>(
          runner.inboxes<my_component>().at(neighbor_id));

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
