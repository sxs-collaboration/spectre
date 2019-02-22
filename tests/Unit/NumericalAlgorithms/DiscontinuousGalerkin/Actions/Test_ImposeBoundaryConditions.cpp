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

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
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
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>

// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"  // for Variables
// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare dg::Actions::ImposeDirichletBoundaryConditions
// IWYU pragma: no_forward_declare dg::Actions::ReceiveDataForFluxes

namespace {
constexpr size_t Dim = 2;

using TemporalId = Tags::TimeId;

struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

struct PrimitiveVar : db::SimpleTag {
  static std::string name() noexcept { return "PrimitiveVar"; }
  using type = Scalar<DataVector>;
};

struct OtherData : db::SimpleTag {
  static std::string name() noexcept { return "OtherData"; }
  using type = Scalar<DataVector>;
};

class NumericalFlux {
 public:
  struct ExtraData : db::SimpleTag {
    static std::string name() noexcept { return "ExtraTag"; }
    using type = tnsr::I<DataVector, 1>;
  };

  using package_tags = tmpl::list<ExtraData, Var>;

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

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct BoundaryCondition {
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& /*x*/,
                                     double /*t*/,
                                     tmpl::list<Var> /*meta*/) const noexcept {
    return tuples::TaggedTuple<Var>{Scalar<DataVector>{{{{30., 40., 50.}}}}};
  }

  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& /*x*/,
                                     double /*t*/,
                                     tmpl::list<PrimitiveVar> /*meta*/) const
      noexcept {
    return tuples::TaggedTuple<Var>{Scalar<DataVector>{{{{15., 20., 25.}}}}};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct BoundaryConditionTag {
  using type = BoundaryCondition;
};

template <bool HasPrimitiveAndConservativeVars>
struct System {
  static constexpr const size_t volume_dim = Dim;
  static constexpr bool is_in_flux_conservative_form = true;
  static constexpr bool has_primitive_and_conservative_vars =
      HasPrimitiveAndConservativeVars;

  using variables_tag = Tags::Variables<tmpl::list<Var>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  struct conservative_from_primitive {
    using return_tags = tmpl::list<Var>;
    using argument_tags = tmpl::list<PrimitiveVar>;

    static void apply(const gsl::not_null<Scalar<DataVector>*> var,
                      const Scalar<DataVector>& primitive_var) {
      get(*var) = 2.0 * get(primitive_var);
    }
  };
};

template <typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;
template <typename Tag>
using interface_compute_tag =
    Tags::InterfaceComputeItem<Tags::InternalDirections<Dim>, Tag>;

template <typename Tag>
using boundary_tag =
    Tags::Interface<Tags::BoundaryDirectionsInterior<Dim>, Tag>;
template <typename Tag>
using boundary_compute_tag =
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>, Tag>;

template <typename Tag>
using external_boundary_tag =
    Tags::Interface<Tags::BoundaryDirectionsExterior<Dim>, Tag>;
template <typename Tag>
using external_boundary_compute_tag =
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<Dim>, Tag>;

template <typename FluxCommTypes>
using mortar_data_tag = typename FluxCommTypes::simple_mortar_data_tag;
template <typename FluxCommTypes>
using LocalMortarData = typename FluxCommTypes::LocalMortarData;
template <typename FluxCommTypes>
using PackagedData = typename FluxCommTypes::PackagedData;
template <typename FluxCommTypes>
using normal_dot_fluxes_tag =
    interface_tag<typename FluxCommTypes::normal_dot_fluxes_tag>;
template <typename FluxCommTypes>
using bdry_normal_dot_fluxes_tag =
    boundary_tag<typename FluxCommTypes::normal_dot_fluxes_tag>;
template <typename FluxCommTypes>
using external_bdry_normal_dot_fluxes_tag =
    external_boundary_tag<typename FluxCommTypes::normal_dot_fluxes_tag>;

using bdry_vars_tag = boundary_tag<Tags::Variables<tmpl::list<Var>>>;
using external_bdry_vars_tag =
    external_boundary_tag<Tags::Variables<tmpl::list<Var>>>;

template <typename FluxCommTypes>
using fluxes_tag = typename FluxCommTypes::FluxesTag;

using other_data_tag = interface_tag<Tags::Variables<tmpl::list<OtherData>>>;
using bdry_other_data_tag =
    boundary_tag<Tags::Variables<tmpl::list<OtherData>>>;
using external_bdry_other_data_tag =
    external_boundary_tag<Tags::Variables<tmpl::list<OtherData>>>;
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, Dim>;
using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>;
using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>;

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list =
      tmpl::list<NumericalFluxTag, BoundaryConditionTag>;
  using action_list =
      tmpl::list<dg::Actions::ImposeDirichletBoundaryConditions<Metavariables>,
                 dg::Actions::ReceiveDataForFluxes<Metavariables>>;
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;

  using simple_tags = db::AddSimpleTags<
      TemporalId, Tags::Next<TemporalId>, Tags::Mesh<Dim>, Tags::Element<Dim>,
      Tags::ElementMap<Dim>, bdry_normal_dot_fluxes_tag<flux_comm_types>,
      bdry_other_data_tag, external_bdry_normal_dot_fluxes_tag<flux_comm_types>,
      external_bdry_other_data_tag, external_bdry_vars_tag,
      mortar_data_tag<flux_comm_types>, mortar_next_temporal_ids_tag,
      mortar_meshes_tag, mortar_sizes_tag>;

  using compute_tags = db::AddComputeTags<
      Tags::Time, Tags::BoundaryDirectionsInterior<Dim>,
      boundary_compute_tag<Tags::Direction<Dim>>,
      boundary_compute_tag<Tags::InterfaceMesh<Dim>>,
      boundary_compute_tag<Tags::UnnormalizedFaceNormal<Dim>>,
      boundary_compute_tag<
          Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
      boundary_compute_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>,
      Tags::BoundaryDirectionsExterior<Dim>,
      external_boundary_compute_tag<Tags::Direction<Dim>>,
      external_boundary_compute_tag<Tags::InterfaceMesh<Dim>>,
      external_boundary_compute_tag<Tags::BoundaryCoordinates<Dim>>,
      external_boundary_compute_tag<Tags::UnnormalizedFaceNormal<Dim>>,
      external_boundary_compute_tag<
          Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
      external_boundary_compute_tag<
          Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;

  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

template <bool HasPrimitiveAndConservativeVars>
struct Metavariables {
  using system = System<HasPrimitiveAndConservativeVars>;
  using component_list = tmpl::list<component<Metavariables>>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag;
  using boundary_condition_tag = BoundaryConditionTag;
  using analytic_solution_tag = boundary_condition_tag;
};

template <typename Component>
using compute_items = typename Component::compute_tags;

template <bool HasConservativeAndPrimitiveVars>
void run_test() {
  using metavariables = Metavariables<HasConservativeAndPrimitiveVars>;
  using flux_comm_types = typename component<metavariables>::flux_comm_types;
  using my_component = component<metavariables>;
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  //      xi      Block       +- xi
  //      |     0   |   1     |
  // eta -+ +-------+-+-+---+ eta
  //        |       |X| |   |
  //        |       +-+-+   |
  //        |       | | |   |
  //        +-------+-+-+---+
  // We run the actions on the indicated element.
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> east_id(1, {{{2, 1}, {1, 0}}});
  const ElementId<2> south_id(1, {{{2, 0}, {1, 1}}});

  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};

  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                 eta_map));

  const auto external_directions = {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()};

  const auto external_mortar_ids = {
      std::make_pair(Direction<2>::lower_eta(),
                     ElementId<2>::external_boundary_id()),
      std::make_pair(Direction<2>::upper_xi(),
                     ElementId<2>::external_boundary_id())};

  const struct {
    std::unordered_map<Direction<2>, Scalar<DataVector>> bdry_fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>> bdry_other_data;
    std::unordered_map<Direction<2>, Scalar<DataVector>> external_bdry_fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>>
        external_bdry_other_data;
    std::unordered_map<Direction<2>, Scalar<DataVector>> external_bdry_vars;
  } data{{{Direction<2>::lower_eta(), Scalar<DataVector>{{{{1., 2., 3.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{21., 22., 23.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{4., 5., 6.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{24., 25., 26.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{7., 8., 9.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{27., 28., 29.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{10., 11., 12.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{30., 31., 32.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{13., 14., 15.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{33., 34., 35.}}}}}}};

  auto start_box = [
    &mesh, &self_id, &west_id, &south_id, &coordmap, &data,
    &external_directions, &external_mortar_ids
  ]() noexcept {
    const Element<2> element(self_id,
                             {{Direction<2>::lower_xi(), {{west_id}, {}}},
                              {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag<flux_comm_types>>
        bdry_normal_dot_fluxes;
    for (const auto& direction : external_directions) {
      auto& flux_vars = bdry_normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<Var>>(flux_vars) = data.bdry_fluxes.at(direction);
    }

    db::item_type<other_data_tag> bdry_other_data;
    for (const auto& direction : external_directions) {
      auto& other_data_vars = bdry_other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) = data.bdry_other_data.at(direction);
    }

    db::item_type<normal_dot_fluxes_tag<flux_comm_types>>
        external_bdry_normal_dot_fluxes;
    for (const auto& direction : external_directions) {
      auto& flux_vars = external_bdry_normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<Var>>(flux_vars) =
          data.external_bdry_fluxes.at(direction);
    }

    db::item_type<other_data_tag> external_bdry_other_data;
    for (const auto& direction : external_directions) {
      auto& other_data_vars = external_bdry_other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) =
          data.external_bdry_other_data.at(direction);
    }

    db::item_type<external_bdry_vars_tag> external_bdry_vars;
    for (const auto& direction : external_directions) {
      auto& vars = external_bdry_vars[direction];
      vars.initialize(3);
      get<Var>(vars) = data.external_bdry_vars.at(direction);
    }

    const Slab slab(1.2, 3.4);
    const Time start = slab.start();
    const Time end = slab.end();

    TimeId initial_time(true, 4, start);
    TimeId next_time(true, 4, end);

    db::item_type<mortar_data_tag<flux_comm_types>> mortar_history{};
    db::item_type<mortar_next_temporal_ids_tag> mortar_next_temporal_ids{};
    db::item_type<mortar_meshes_tag> mortar_meshes{};
    db::item_type<mortar_sizes_tag> mortar_sizes{};
    for (const auto& mortar_id : external_mortar_ids) {
      mortar_history.insert({mortar_id, {}});
      mortar_next_temporal_ids.insert({mortar_id, initial_time});
      mortar_meshes.insert({mortar_id, mesh.slice_away(0)});
      mortar_sizes.insert({mortar_id, {{Spectral::MortarSize::Full}}});
    }

    return db::create<
        db::AddSimpleTags<
            TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>, Tags::Element<2>,
            Tags::ElementMap<2>, bdry_normal_dot_fluxes_tag<flux_comm_types>,
            bdry_other_data_tag,
            external_bdry_normal_dot_fluxes_tag<flux_comm_types>,
            external_bdry_other_data_tag, external_bdry_vars_tag,
            mortar_data_tag<flux_comm_types>, mortar_next_temporal_ids_tag,
            mortar_meshes_tag, mortar_sizes_tag>,
        compute_items<my_component>>(
        initial_time, next_time, mesh, element, std::move(map),
        std::move(bdry_normal_dot_fluxes), std::move(bdry_other_data),
        std::move(external_bdry_normal_dot_fluxes),
        std::move(external_bdry_other_data), std::move(external_bdry_vars),
        std::move(mortar_history), std::move(mortar_next_temporal_ids),
        std::move(mortar_meshes), std::move(mortar_sizes));
  }
  ();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          my_component>;
  typename MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id, std::move(start_box));

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {NumericalFlux{}, BoundaryCondition{}}, std::move(dist_objects)};

  using initial_databox_type = db::compute_databox_type<tmpl::append<
      db::AddSimpleTags<
          TemporalId, Tags::Next<TemporalId>, Tags::Mesh<2>, Tags::Element<2>,
          Tags::ElementMap<2>, bdry_normal_dot_fluxes_tag<flux_comm_types>,
          bdry_other_data_tag,
          external_bdry_normal_dot_fluxes_tag<flux_comm_types>,
          external_bdry_other_data_tag, external_bdry_vars_tag,
          mortar_data_tag<flux_comm_types>, mortar_next_temporal_ids_tag,
          mortar_meshes_tag, mortar_sizes_tag>,
      compute_items<my_component>>>;

  runner.template next_action<my_component>(self_id);

  CHECK(runner.template is_ready<my_component>(self_id));

  // Check that BC's were indeed applied.
  const auto& external_vars = db::get<external_bdry_vars_tag>(
      runner.template algorithms<my_component>()
          .at(self_id)
          .template get_databox<initial_databox_type>());

  db::item_type<external_bdry_vars_tag> expected_vars{};
  for (const auto& direction : external_directions) {
    expected_vars[direction].initialize(3);
  }
  get<Var>(expected_vars[Direction<2>::lower_eta()]) =
      Scalar<DataVector>({{{30., 40., 50.}}});
  get<Var>(expected_vars[Direction<2>::upper_xi()]) =
      Scalar<DataVector>({{{30., 40., 50.}}});
  CHECK(external_vars == expected_vars);

  // ReceiveDataForFluxes
  runner.template next_action<my_component>(self_id);
  const auto& self_box = runner.template algorithms<my_component>()
                             .at(self_id)
                             .template get_databox<initial_databox_type>();

  auto mortar_history = serialize_and_deserialize(
      db::get<mortar_data_tag<flux_comm_types>>(self_box));
  CHECK(mortar_history.size() == 2);
  const auto check_mortar = [&mortar_history](
      const std::pair<Direction<2>, ElementId<2>>& mortar_id,
      const Scalar<DataVector>& local_flux,
      const Scalar<DataVector>& remote_flux,
      const Scalar<DataVector>& local_other,
      const Scalar<DataVector>& remote_other,
      const tnsr::i<DataVector, 2>& local_normal,
      const tnsr::i<DataVector, 2>& remote_normal) noexcept {
    LocalMortarData<flux_comm_types> local_mortar_data(3);
    get<Tags::NormalDotFlux<Var>>(local_mortar_data) = local_flux;
    const auto magnitude_local_normal = magnitude(local_normal);
    auto normalized_local_normal = local_normal;
    for (auto& x : normalized_local_normal) {
      x /= get(magnitude_local_normal);
    }
    PackagedData<flux_comm_types> local_packaged(3);
    NumericalFlux{}.package_data(&local_packaged, local_flux, local_other,
                                 normalized_local_normal);
    local_mortar_data.assign_subset(local_packaged);

    const auto magnitude_remote_normal = magnitude(remote_normal);
    auto normalized_remote_normal = remote_normal;
    for (auto& x : normalized_remote_normal) {
      x /= get(magnitude_remote_normal);
    }
    PackagedData<flux_comm_types> remote_packaged(3);
    NumericalFlux{}.package_data(&remote_packaged, remote_flux, remote_other,
                                 normalized_remote_normal);

    const auto result = mortar_history.at(mortar_id).extract();

    CHECK(result.first.mortar_data == local_mortar_data);
    CHECK(result.first.magnitude_of_face_normal == magnitude_local_normal);
    CHECK(result.second == remote_packaged);
  };

  check_mortar(
      std::make_pair(Direction<2>::lower_eta(),
                     ElementId<2>::external_boundary_id()),
      data.bdry_fluxes.at(Direction<2>::lower_eta()),
      data.external_bdry_fluxes.at(Direction<2>::lower_eta()),
      data.bdry_other_data.at(Direction<2>::lower_eta()),
      data.external_bdry_other_data.at(Direction<2>::lower_eta()),
      tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, 1.0}}}},
      tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, -1.0}}}});

  check_mortar(
      std::make_pair(Direction<2>::upper_xi(),
                     ElementId<2>::external_boundary_id()),
      data.bdry_fluxes.at(Direction<2>::upper_xi()),
      data.external_bdry_fluxes.at(Direction<2>::upper_xi()),
      data.bdry_other_data.at(Direction<2>::upper_xi()),
      data.external_bdry_other_data.at(Direction<2>::upper_xi()),
      tnsr::i<DataVector, 2>{{{DataVector{3, 2.0}, DataVector{3, 0.0}}}},
      tnsr::i<DataVector, 2>{{{DataVector{3, -2.0}, DataVector{3, 0.0}}}});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.BoundaryConditions",
                  "[Unit][NumericalAlgorithms][Actions]") {
  run_test<false>();
  run_test<true>();
}
