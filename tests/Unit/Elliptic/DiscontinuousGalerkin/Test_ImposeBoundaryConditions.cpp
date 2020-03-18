// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/DiscontinuousGalerkin/FluxCommunication.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <boost/functional/hash/extensions.hpp>

// IWYU pragma: no_include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
// IWYU pragma: no_include "Parallel/PupStlCpp11.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
// IWYU pragma: no_forward_declare dg::Actions::ReceiveDataForFluxes

// IWYU pragma: no_include <boost/variant/get.hpp>

// Note: Most of this test is adapted from:
// `NumericalAlgorithms/DiscontinuousGalerkin/Actions/
// Test_ImposeBoundaryConditions.cpp`

namespace {
constexpr size_t Dim = 2;

struct TemporalId : db::SimpleTag {
  using type = int;
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};

using field_tag = ScalarField;
using vars_tag = Tags::Variables<tmpl::list<field_tag>>;

struct OtherData : db::SimpleTag {
  using type = Scalar<DataVector>;
};

class NumericalFlux {
 public:
  struct ExtraData : db::SimpleTag {
    using type = tnsr::I<DataVector, 1>;
  };

  using argument_tags =
      tmpl::list<Tags::NormalDotFlux<field_tag>, OtherData,
                 Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  using package_tags = tmpl::list<ExtraData, field_tag>;

  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& scalar_field_flux,
                    const Scalar<DataVector>& other_data,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) const noexcept {
    get(get<field_tag>(*packaged_data)) = 10. * get(scalar_field_flux);
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
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = vars_tag;
  using primal_variables = tmpl::list<ScalarField>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <typename Tag>
using interface_tag =
    domain::Tags::Interface<domain::Tags::InternalDirections<Dim>, Tag>;
template <typename Tag>
using interface_compute_tag =
    domain::Tags::InterfaceCompute<domain::Tags::InternalDirections<Dim>, Tag>;

template <typename Tag>
using boundary_tag =
    domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>, Tag>;
template <typename Tag>
using boundary_compute_tag = domain::Tags::InterfaceCompute<
    domain::Tags::BoundaryDirectionsInterior<Dim>, Tag>;

template <typename Tag>
using exterior_boundary_tag =
    domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>, Tag>;
template <typename Tag>
using exterior_boundary_compute_tag = domain::Tags::InterfaceCompute<
    domain::Tags::BoundaryDirectionsExterior<Dim>, Tag>;

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
using exterior_bdry_normal_dot_fluxes_tag =
    exterior_boundary_tag<typename FluxCommTypes::normal_dot_fluxes_tag>;

using bdry_vars_tag = boundary_tag<vars_tag>;
using exterior_bdry_vars_tag = exterior_boundary_tag<vars_tag>;

template <typename FluxCommTypes>
using fluxes_tag = typename FluxCommTypes::FluxesTag;

using other_data_tag = interface_tag<Tags::Variables<tmpl::list<OtherData>>>;
using bdry_other_data_tag =
    boundary_tag<Tags::Variables<tmpl::list<OtherData>>>;
using exterior_bdry_other_data_tag =
    exterior_boundary_tag<Tags::Variables<tmpl::list<OtherData>>>;
using mortar_next_temporal_ids_tag = Tags::Mortars<Tags::Next<TemporalId>, Dim>;
using mortar_meshes_tag = Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>;
using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>;

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<NumericalFluxTag>;

  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
  using simple_tags = db::AddSimpleTags<
      TemporalId, Tags::Next<TemporalId>, domain::Tags::Mesh<Dim>,
      domain::Tags::Element<Dim>, domain::Tags::ElementMap<Dim>, bdry_vars_tag,
      bdry_normal_dot_fluxes_tag<flux_comm_types>, bdry_other_data_tag,
      exterior_bdry_normal_dot_fluxes_tag<flux_comm_types>,
      exterior_bdry_other_data_tag, exterior_bdry_vars_tag,
      mortar_data_tag<flux_comm_types>, mortar_next_temporal_ids_tag,
      mortar_meshes_tag, mortar_sizes_tag>;

  using compute_tags = db::AddComputeTags<
      domain::Tags::BoundaryDirectionsInterior<Dim>,
      boundary_compute_tag<domain::Tags::Direction<Dim>>,
      boundary_compute_tag<domain::Tags::InterfaceMesh<Dim>>,
      boundary_compute_tag<domain::Tags::UnnormalizedFaceNormalCompute<Dim>>,
      boundary_compute_tag<
          Tags::EuclideanMagnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      boundary_compute_tag<
          Tags::NormalizedCompute<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      domain::Tags::BoundaryDirectionsExterior<Dim>,
      exterior_boundary_compute_tag<domain::Tags::Direction<Dim>>,
      exterior_boundary_compute_tag<domain::Tags::InterfaceMesh<Dim>>,
      exterior_boundary_compute_tag<domain::Tags::BoundaryCoordinates<Dim>>,
      exterior_boundary_compute_tag<
          domain::Tags::UnnormalizedFaceNormalCompute<Dim>>,
      exterior_boundary_compute_tag<
          Tags::EuclideanMagnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      exterior_boundary_compute_tag<
          Tags::NormalizedCompute<domain::Tags::UnnormalizedFaceNormal<Dim>>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              elliptic::dg::Actions::
                  ImposeHomogeneousDirichletBoundaryConditions<Metavariables>,
              dg::Actions::ReceiveDataForFluxes<Metavariables>>>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  using temporal_id = TemporalId;

  using normal_dot_numerical_flux = NumericalFluxTag;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Component>
using compute_items = typename Component::compute_tags;

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.BoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  using flux_comm_types = typename ElementArray<Metavariables>::flux_comm_types;
  using my_component = ElementArray<Metavariables>;
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  // Domain decomposition:
  //
  //  Block 0   Block 1
  // +-------+ +-------+-> xi
  // |       | |   X   |
  // |       | +-------+
  // |       | |       |
  // +-------+ +-------+
  //                   v eta

  // `self_id` is the Element indicated by the X in the diagram above
  const ElementId<2> self_id(1, {{{0, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> south_id(1, {{{0, 0}, {1, 1}}});

  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};

  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(xi_map, eta_map));
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));

  const auto external_directions = {Direction<2>::lower_eta(),
                                    Direction<2>::upper_xi()};

  const auto external_mortar_ids = {
      std::make_pair(Direction<2>::lower_eta(),
                     ElementId<2>::external_boundary_id()),
      std::make_pair(Direction<2>::upper_xi(),
                     ElementId<2>::external_boundary_id())};

  const struct {
    std::unordered_map<Direction<2>, Scalar<DataVector>> bdry_vars;
    std::unordered_map<Direction<2>, Scalar<DataVector>> bdry_fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>> bdry_other_data;
    std::unordered_map<Direction<2>, Scalar<DataVector>> exterior_bdry_fluxes;
    std::unordered_map<Direction<2>, Scalar<DataVector>>
        exterior_bdry_other_data;
    std::unordered_map<Direction<2>, Scalar<DataVector>> exterior_bdry_vars;
  } data{{{Direction<2>::lower_eta(), Scalar<DataVector>{{{{-1., -2., -3.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{-4., -5., -6.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{1., 2., 3.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{21., 22., 23.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{4., 5., 6.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{24., 25., 26.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{7., 8., 9.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{27., 28., 29.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{10., 11., 12.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{30., 31., 32.}}}}}},
         {{Direction<2>::lower_eta(), Scalar<DataVector>{{{{13., 14., 15.}}}}},
          {Direction<2>::upper_xi(), Scalar<DataVector>{{{{33., 34., 35.}}}}}}};

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  MockRuntimeSystem runner{{NumericalFlux{}}};
  {
    const Element<2> element(self_id,
                             {{Direction<2>::lower_xi(), {{west_id}, {}}},
                              {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto element_map =
        ElementMap<2, Frame::Inertial>(self_id, coord_map->get_clone());

    db::item_type<bdry_vars_tag> bdry_vars;
    for (const auto& direction : external_directions) {
      auto& flux_vars = bdry_vars[direction];
      flux_vars.initialize(3);
      get<field_tag>(flux_vars) = data.bdry_vars.at(direction);
    }

    db::item_type<normal_dot_fluxes_tag<flux_comm_types>>
        bdry_normal_dot_fluxes;
    for (const auto& direction : external_directions) {
      auto& flux_vars = bdry_normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<field_tag>>(flux_vars) =
          data.bdry_fluxes.at(direction);
    }

    db::item_type<other_data_tag> bdry_other_data;
    for (const auto& direction : external_directions) {
      auto& other_data_vars = bdry_other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) = data.bdry_other_data.at(direction);
    }

    db::item_type<normal_dot_fluxes_tag<flux_comm_types>>
        exterior_bdry_normal_dot_fluxes;
    for (const auto& direction : external_directions) {
      auto& flux_vars = exterior_bdry_normal_dot_fluxes[direction];
      flux_vars.initialize(3);
      get<Tags::NormalDotFlux<field_tag>>(flux_vars) =
          data.exterior_bdry_fluxes.at(direction);
    }

    db::item_type<other_data_tag> exterior_bdry_other_data;
    for (const auto& direction : external_directions) {
      auto& other_data_vars = exterior_bdry_other_data[direction];
      other_data_vars.initialize(3);
      get<OtherData>(other_data_vars) =
          data.exterior_bdry_other_data.at(direction);
    }

    db::item_type<exterior_bdry_vars_tag> exterior_bdry_vars;
    for (const auto& direction : external_directions) {
      auto& vars = exterior_bdry_vars[direction];
      vars.initialize(3);
      get<field_tag>(vars) = data.exterior_bdry_vars.at(direction);
    }

    const int initial_temporal_id = 0;
    const int next_temporal_id = 1;

    db::item_type<mortar_data_tag<flux_comm_types>> mortar_history{};
    db::item_type<mortar_next_temporal_ids_tag> mortar_next_temporal_ids{};
    db::item_type<mortar_meshes_tag> mortar_meshes{};
    db::item_type<mortar_sizes_tag> mortar_sizes{};
    for (const auto& mortar_id : external_mortar_ids) {
      mortar_history.insert({mortar_id, {}});
      mortar_next_temporal_ids.insert({mortar_id, initial_temporal_id});
      mortar_meshes.insert({mortar_id, mesh.slice_away(0)});
      mortar_sizes.insert({mortar_id, {{Spectral::MortarSize::Full}}});
    }

    ActionTesting::emplace_component_and_initialize<my_component>(
        &runner, self_id,
        {initial_temporal_id, next_temporal_id, mesh, element,
         std::move(element_map), std::move(bdry_vars),
         std::move(bdry_normal_dot_fluxes), std::move(bdry_other_data),
         std::move(exterior_bdry_normal_dot_fluxes),
         std::move(exterior_bdry_other_data), std::move(exterior_bdry_vars),
         std::move(mortar_history), std::move(mortar_next_temporal_ids),
         std::move(mortar_meshes), std::move(mortar_sizes)});
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  using additional_tags = tmpl::append<typename my_component::simple_tags,
                                       typename my_component::compute_tags>;

  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);

  CHECK(ActionTesting::is_ready<my_component>(runner, self_id));

  // Check that BC's were indeed applied.
  const auto& exterior_vars =
      ActionTesting::get_databox_tag<my_component, exterior_bdry_vars_tag>(
          runner, self_id);

  db::item_type<exterior_bdry_vars_tag> expected_vars{};
  for (const auto& direction : external_directions) {
    expected_vars[direction].initialize(3);
  }
  get<field_tag>(expected_vars[Direction<2>::lower_eta()]) =
      Scalar<DataVector>({{{1., 2., 3.}}});
  get<field_tag>(expected_vars[Direction<2>::upper_xi()]) =
      Scalar<DataVector>({{{4., 5., 6.}}});
  CHECK(exterior_vars == expected_vars);

  // next_action is ReceiveDataForFluxes
  ActionTesting::next_action<my_component>(make_not_null(&runner), self_id);
  const auto& self_box =
      ActionTesting::get_databox<my_component, additional_tags>(runner,
                                                                self_id);

  auto mortar_history = serialize_and_deserialize(
      db::get<mortar_data_tag<flux_comm_types>>(self_box));
  CHECK(mortar_history.size() == 2);
  const auto check_mortar = [&mortar_history](
      const std::pair<Direction<2>, ElementId<2>>& mortar_id,
      const Scalar<DataVector>& local_flux,
      const Scalar<DataVector>& remote_flux,
      const Scalar<DataVector>& local_other,
      const Scalar<DataVector>& remote_other,
      const tnsr::i<DataVector, 2>& normalized_local_normal,
      const tnsr::i<DataVector, 2>& normalized_remote_normal) noexcept {
    LocalMortarData<flux_comm_types> local_mortar_data(3);
    get<Tags::NormalDotFlux<field_tag>>(local_mortar_data) = local_flux;
    PackagedData<flux_comm_types> local_packaged(3);
    NumericalFlux{}.package_data(&local_packaged, local_flux, local_other,
                                 normalized_local_normal);
    local_mortar_data.assign_subset(local_packaged);

    PackagedData<flux_comm_types> remote_packaged(3);
    NumericalFlux{}.package_data(&remote_packaged, remote_flux, remote_other,
                                 normalized_remote_normal);

    const auto result = mortar_history.at(mortar_id).extract();

    CHECK(result.first.mortar_data == local_mortar_data);
    CHECK(result.second == remote_packaged);
  };

  check_mortar(
      std::make_pair(Direction<2>::lower_eta(),
                     ElementId<2>::external_boundary_id()),
      data.bdry_fluxes.at(Direction<2>::lower_eta()),
      data.exterior_bdry_fluxes.at(Direction<2>::lower_eta()),
      data.bdry_other_data.at(Direction<2>::lower_eta()),
      data.exterior_bdry_other_data.at(Direction<2>::lower_eta()),
      tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, 1.0}}}},
      tnsr::i<DataVector, 2>{{{DataVector{3, 0.0}, DataVector{3, -1.0}}}});

  check_mortar(
      std::make_pair(Direction<2>::upper_xi(),
                     ElementId<2>::external_boundary_id()),
      data.bdry_fluxes.at(Direction<2>::upper_xi()),
      data.exterior_bdry_fluxes.at(Direction<2>::upper_xi()),
      data.bdry_other_data.at(Direction<2>::upper_xi()),
      data.exterior_bdry_other_data.at(Direction<2>::upper_xi()),
      tnsr::i<DataVector, 2>{{{DataVector{3, 1.0}, DataVector{3, 0.0}}}},
      tnsr::i<DataVector, 2>{{{DataVector{3, -1.0}, DataVector{3, 0.0}}}});
}

}  // namespace
