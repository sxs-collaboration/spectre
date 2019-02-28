// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include <unordered_map>

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace {
struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; }
  using type = size_t;
  template <typename Tag>
  using step_prefix = Tags::dt<Tag>;
};

struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

class NumericalFlux {
 public:
  struct ExtraData : db::SimpleTag {
    static std::string name() noexcept { return "ExtraData"; }
    using type = tnsr::I<DataVector, 1>;
  };

  using package_tags = tmpl::list<ExtraData, Var>;

  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux_var,
                  const tnsr::I<DataVector, 1>& extra_data_local,
                  const Scalar<DataVector>& var_local,
                  const tnsr::I<DataVector, 1>& extra_data_remote,
                  const Scalar<DataVector>& var_remote) const {
    get(*numerical_flux_var) = 10. * get(var_local) + 1000. * get(var_remote);
    CHECK(get<0>(extra_data_local) == 3. * get<0>(extra_data_remote));
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <typename Flux>
struct NumericalFluxTag {
  using type = Flux;
};

template <size_t Dim>
struct System {
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

template <size_t Dim, typename Flux, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<NumericalFluxTag<Flux>>;
  using action_list = tmpl::list<dg::Actions::ApplyFluxes>;
  using initial_databox = db::compute_databox_type<
      tmpl::list<Tags::Mesh<Dim>, Tags::Coordinates<Dim, Frame::Logical>,
                 Tags::Mortars<Tags::Mesh<Dim - 1>, Dim>,
                 Tags::Mortars<Tags::MortarSize<Dim - 1>, Dim>,
                 Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>,
                 typename dg::FluxCommunicationTypes<
                     Metavariables>::simple_mortar_data_tag>>;
};

template <size_t Dim, typename Flux>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<component<Dim, Flux, Metavariables>>;
  using temporal_id = TemporalId;
  static constexpr bool local_time_stepping = false;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag<Flux>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.ApplyFluxes",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using flux_comm_types =
      dg::FluxCommunicationTypes<Metavariables<2, NumericalFlux>>;
  using my_component =
      component<2, NumericalFlux, Metavariables<2, NumericalFlux>>;
  using LocalData = typename flux_comm_types::LocalData;
  using PackagedData = typename flux_comm_types::PackagedData;
  using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
  using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<1>, 2>;
  using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<1>, 2>;

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const ElementId<2> id(0);

  typename mortar_meshes_tag::type mortar_meshes{
      {{Direction<2>::upper_xi(), id}, mesh.slice_away(0)},
      {{Direction<2>::lower_eta(), id}, mesh.slice_away(1)}};
  typename mortar_sizes_tag::type mortar_sizes{
      {{Direction<2>::upper_xi(), id}, {{Spectral::MortarSize::Full}}},
      {{Direction<2>::lower_eta(), id}, {{Spectral::MortarSize::Full}}}};

  const Variables<tmpl::list<Tags::dt<Var>>> initial_dt(
      mesh.number_of_grid_points(), 5.);

  const auto make_mortar_data = [](
      const Scalar<DataVector>& local_var,
      const Scalar<DataVector>& remote_var,
      const tnsr::I<DataVector, 1>& local_extra_data,
      const tnsr::I<DataVector, 1>& remote_extra_data,
      const Scalar<DataVector>& local_flux,
      const Scalar<DataVector>& local_magnitude_face_normal) noexcept {
    dg::SimpleBoundaryData<size_t, LocalData, PackagedData> data;

    LocalData local_data{};
    local_data.mortar_data.initialize(3);
    get<Tags::NormalDotFlux<Var>>(local_data.mortar_data) = local_flux;
    get<NumericalFlux::ExtraData>(local_data.mortar_data) = local_extra_data;
    get<Var>(local_data.mortar_data) = local_var;
    local_data.magnitude_of_face_normal = local_magnitude_face_normal;
    data.local_insert(0, std::move(local_data));

    PackagedData remote_data(3);
    get<NumericalFlux::ExtraData>(remote_data) = remote_extra_data;
    get<Var>(remote_data) = remote_var;
    data.remote_insert(0, std::move(remote_data));

    return data;
  };

  mortar_data_tag::type mortar_data{
      {{Direction<2>::upper_xi(), id},
       make_mortar_data(
           Scalar<DataVector>{{{{1., 2., 3.}}}},         // local var
           Scalar<DataVector>{{{{7., 8., 9.}}}},         // remote var
           tnsr::I<DataVector, 1>{{{{15., 18., 21.}}}},  // local ExtraData
           tnsr::I<DataVector, 1>{{{{5., 6., 7.}}}},     // remote ExtraData
           Scalar<DataVector>{{{{-1., -3., -5.}}}},      // flux
           Scalar<DataVector>{{{{2., 2., 2.}}}})},       // normal magnitude
      {{Direction<2>::lower_eta(), id},
       make_mortar_data(
           Scalar<DataVector>{{{{4., 5., 6.}}}},         // local var
           Scalar<DataVector>{{{{10., 11., 12.}}}},      // remote var
           tnsr::I<DataVector, 1>{{{{12., 15., 18.}}}},  // local ExtraData
           tnsr::I<DataVector, 1>{{{{4., 5., 6.}}}},     // remote ExtraData
           Scalar<DataVector>{{{{-2., -4., -6.}}}},      // flux
           Scalar<DataVector>{{{{3., 3., 3.}}}})}};      // normal magnitude

  using simple_tags =
      db::AddSimpleTags<Tags::Mesh<2>, Tags::Coordinates<2, Frame::Logical>,
                        mortar_meshes_tag, mortar_sizes_tag,
                        Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>,
                        mortar_data_tag>;

  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<2, NumericalFlux>>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<
          component<2, NumericalFlux, Metavariables<2, NumericalFlux>>>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(id, db::create<simple_tags>(mesh, logical_coordinates(mesh),
                                           std::move(mortar_meshes),
                                           std::move(mortar_sizes), initial_dt,
                                           std::move(mortar_data)));
  MockRuntimeSystem runner{{NumericalFlux{}}, std::move(dist_objects)};

  runner.next_action<my_component>(id);

  // F* - F = 10 * local_var + 1000 * remote_var - local_flux
  const DataVector xi_flux = {0., 0., 7011.,
                              0., 0., 8023.,
                              0., 0., 9035.};
  const DataVector eta_flux = {10042., 11054., 12066.,
                               0., 0., 0.,
                               0., 0., 0.};
  // These factors are (see Kopriva 8.42)
  // -3[based on extents] * magnitude_of_normal
  CHECK_ITERABLE_APPROX(
      get(db::get<Tags::dt<Var>>(
          runner.algorithms<my_component>()
              .at(id)
              .get_databox<db::compute_databox_type<simple_tags>>())),
      get(get<Tags::dt<Var>>(initial_dt)) - 6. * xi_flux - 9. * eta_flux);
}

namespace {
class RefinementNumericalFlux {
 public:
  using package_tags = tmpl::list<Tags::NormalDotFlux<Var>>;

  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux_var,
                  const Scalar<DataVector>& local,
                  const Scalar<DataVector>& remote) const {
    get(*numerical_flux_var) = 0.5 * (get(local) - get(remote));
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

Scalar<DataVector> flux1(
    const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
  return Scalar<DataVector>(exp(get<0>(coords)));
}
Scalar<DataVector> flux2(
    const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
  return Scalar<DataVector>(log(get<1>(coords) + 2.));
}

// Map is
// (xi, eta, zeta) -> ((2. + sin(square(eta) * zeta)) * (xi - 1), eta, zeta)
Scalar<DataVector> determinant_of_jacobian1(
    const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
  return Scalar<DataVector>(
      1. / (2. + sin(square(get<1>(coords)) * get<2>(coords))));
}
Scalar<DataVector> magnitude_of_face_normal1(
    const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
  return Scalar<DataVector>(2. + sin(square(get<0>(coords)) * get<1>(coords)));
}
// Map is the identity
Scalar<DataVector> determinant_of_jacobian2(
    const tnsr::I<DataVector, 3, Frame::Logical>& coords) noexcept {
  return make_with_value<Scalar<DataVector>>(coords, 1.);
}
Scalar<DataVector> magnitude_of_face_normal2(
    const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
  return make_with_value<Scalar<DataVector>>(coords, 1.);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DG.Actions.ApplyFluxes.p-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<3, RefinementNumericalFlux>;
  using my_component = component<3, RefinementNumericalFlux, metavariables>;
  using flux_comm_types = dg::FluxCommunicationTypes<metavariables>;
  using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
  using LocalData = typename flux_comm_types::LocalData;
  using PackagedData = typename flux_comm_types::PackagedData;
  using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<2>, 3>;
  using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<2>, 3>;
  using dt_variables_tag =
      db::add_tag_prefix<Tags::dt, typename System<3>::variables_tag>;

  using simple_tags =
      db::AddSimpleTags<Tags::Mesh<3>, Tags::Coordinates<3, Frame::Logical>,
                        mortar_meshes_tag, mortar_sizes_tag, dt_variables_tag,
                        mortar_data_tag>;

  const ElementId<3> id_0(0);
  const ElementId<3> id_1(1);
  const auto direction = Direction<3>::upper_xi();

  const auto make_initial_box = [](
      const std::array<size_t, 3>& extents) noexcept {
    const Mesh<3> mesh(extents, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    return db::create<simple_tags>(
        mesh, logical_coordinates(mesh), typename mortar_meshes_tag::type{},
        typename mortar_sizes_tag::type{},
        typename dt_variables_tag::type(mesh.number_of_grid_points(), 0.),
        typename mortar_data_tag::type{});
  };

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<
          component<3, RefinementNumericalFlux, metavariables>>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(id_0, make_initial_box({{3, 4, 5}}));
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(id_1, make_initial_box({{4, 2, 6}}));
  MockRuntimeSystem runner{{RefinementNumericalFlux{}},
                           std::move(dist_objects)};

  auto& box1 = runner.algorithms<my_component>()
                   .at(id_0)
                   .get_databox<db::compute_databox_type<simple_tags>>();
  auto& box2 = runner.algorithms<my_component>()
                   .at(id_1)
                   .get_databox<db::compute_databox_type<simple_tags>>();

  const auto set_boundary_data =
      [&direction, &id_0 ](const auto local, const auto remote, const auto flux,
                           const auto magnitude_of_face_normal) noexcept {
    const auto& volume_mesh = get<Tags::Mesh<3>>(*local);
    const auto face_mesh =
        get<Tags::Mesh<3>>(*local).slice_away(direction.dimension());
    const auto remote_mesh =
        get<Tags::Mesh<3>>(*remote).slice_away(direction.dimension());
    const auto mortar_mesh = dg::mortar_mesh(face_mesh, remote_mesh);
    const auto face_coords = logical_coordinates(face_mesh);
    const auto mortar_coords = logical_coordinates(mortar_mesh);

    LocalData local_data{};

    // The flux on the mortar mesh should be exactly representable on
    // the face mesh.
    local_data.mortar_data.initialize(face_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(local_data.mortar_data) = flux(face_coords);
    local_data.mortar_data =
        dg::project_to_mortar(local_data.mortar_data, face_mesh, mortar_mesh,
                              make_array<2>(Spectral::MortarSize::Full));

    local_data.magnitude_of_face_normal = magnitude_of_face_normal(face_coords);
    PackagedData remote_data = local_data.mortar_data;

    Variables<tmpl::list<Tags::NormalDotFlux<Var>>> uncoupled_normal_dot_flux(
        face_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(uncoupled_normal_dot_flux) =
        flux(face_coords);
    const Variables<tmpl::list<Tags::dt<Var>>> uncoupled_lifted_flux(
        dg::lift_flux(uncoupled_normal_dot_flux,
                      volume_mesh.extents(direction.dimension()),
                      local_data.magnitude_of_face_normal));

    db::mutate<dt_variables_tag, mortar_meshes_tag, mortar_sizes_tag,
               mortar_data_tag>(
        local,
        [
          &direction, &id_0, &local_data, &mortar_mesh, &uncoupled_lifted_flux,
          &volume_mesh
        ](const gsl::not_null<db::item_type<dt_variables_tag>*> dt_variables,
          const gsl::not_null<db::item_type<mortar_meshes_tag>*> mortar_meshes,
          const gsl::not_null<db::item_type<mortar_sizes_tag>*> mortar_sizes,
          const gsl::not_null<db::item_type<mortar_data_tag>*>
              mortar_data) noexcept {
          const auto mortar_id = std::make_pair(direction, id_0);
          add_slice_to_data(
              dt_variables, uncoupled_lifted_flux, volume_mesh.extents(),
              direction.dimension(),
              index_to_slice_at(volume_mesh.extents(), direction));
          (*mortar_meshes)[mortar_id] = mortar_mesh;
          mortar_sizes->insert(
              {mortar_id,
               {{Spectral::MortarSize::Full, Spectral::MortarSize::Full}}});
          (*mortar_data)[mortar_id].local_insert(0, std::move(local_data));
        });
    db::mutate<mortar_data_tag>(
        remote, [&direction, &id_0, &remote_data ](
                    const gsl::not_null<db::item_type<mortar_data_tag>*>
                        mortar_data) noexcept {
          const auto mortar_id = std::make_pair(direction, id_0);
          (*mortar_data)[mortar_id].remote_insert(0, std::move(remote_data));
        });
  };
  set_boundary_data(make_not_null(&box1), make_not_null(&box2), flux1,
                    magnitude_of_face_normal1);
  set_boundary_data(make_not_null(&box2), make_not_null(&box1), flux2,
                    magnitude_of_face_normal2);

  runner.next_action<my_component>(id_0);
  runner.next_action<my_component>(id_1);

  const auto& out_box1 =
      runner.algorithms<my_component>()
          .at(id_0)
          .get_databox<db::compute_databox_type<simple_tags>>();
  const auto& out_box2 =
      runner.algorithms<my_component>()
          .at(id_1)
          .get_databox<db::compute_databox_type<simple_tags>>();

  // Check that the operation was conservative.
  const double average_dt1 = definite_integral(
      get(get<Tags::dt<Var>>(out_box1)) *
          get(determinant_of_jacobian1(
              logical_coordinates(get<Tags::Mesh<3>>(out_box1)))),
      get<Tags::Mesh<3>>(out_box1));
  const double average_dt2 = definite_integral(
      get(get<Tags::dt<Var>>(out_box2)) *
          get(determinant_of_jacobian2(
              logical_coordinates(get<Tags::Mesh<3>>(out_box2)))),
      get<Tags::Mesh<3>>(out_box2));
  CHECK(average_dt1 == approx(-average_dt2));
}

SPECTRE_TEST_CASE(
    "Unit.DG.Actions.ApplyFluxes.h-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  using Spectral::MortarSize;

  using metavariables = Metavariables<3, RefinementNumericalFlux>;
  using my_component = component<3, RefinementNumericalFlux, metavariables>;
  using flux_comm_types = dg::FluxCommunicationTypes<metavariables>;
  using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
  using LocalData = typename flux_comm_types::LocalData;
  using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<2>, 3>;
  using mortar_sizes_tag = Tags::Mortars<Tags::MortarSize<2>, 3>;
  using dt_variables_tag =
      db::add_tag_prefix<Tags::dt, typename System<3>::variables_tag>;

  const Mesh<3> mesh(5, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const ElementId<3> self_id(10);
  const auto direction = Direction<3>::upper_xi();

  const auto face_mesh = mesh.slice_away(direction.dimension());
  const auto face_coords = logical_coordinates(face_mesh);
  Variables<tmpl::list<Tags::NormalDotFlux<Var>>> initial_flux(
      face_mesh.number_of_grid_points());
  get<Tags::NormalDotFlux<Var>>(initial_flux) = flux1(face_coords);
  typename dt_variables_tag::type initial_dt_vars(mesh.number_of_grid_points(),
                                                  0.);
  add_slice_to_data(
      make_not_null(&initial_dt_vars),
      typename dt_variables_tag::type(dg::lift_flux(
          std::move(initial_flux), mesh.extents(direction.dimension()),
          magnitude_of_face_normal1(face_coords))),
      mesh.extents(), direction.dimension(),
      index_to_slice_at(mesh.extents(), direction));

  // The coordinates are stored for use in integration later.  They
  // are not needed for the action.

  using simple_tags =
      db::AddSimpleTags<Tags::Mesh<3>, Tags::Coordinates<3, Frame::Logical>,
                        mortar_meshes_tag, mortar_sizes_tag, dt_variables_tag,
                        mortar_data_tag>;
  using db_type = db::compute_databox_type<simple_tags>;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<
          component<3, RefinementNumericalFlux, metavariables>>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id,
               db::create<simple_tags>(mesh, logical_coordinates(mesh),
                                       typename mortar_meshes_tag::type{},
                                       typename mortar_sizes_tag::type{},
                                       std::move(initial_dt_vars),
                                       typename mortar_data_tag::type{}));

  const auto add_neighbor =
      [&direction, &face_coords, &mesh, &self_id, &dist_objects ](
          const std::array<MortarSize, 2>& mortar_size,
          const std::array<double, 2>& center,
          const std::array<double, 2>& half_width,
          const ElementId<3>& neighbor_id) noexcept {
    const auto mortar_id_in_self = std::make_pair(direction, neighbor_id);
    const auto mortar_id_in_neighbor = std::make_pair(direction, self_id);

    const auto mortar_mesh = mesh.slice_away(direction.dimension());
    auto mortar_coords = face_coords;
    get<0>(mortar_coords) = half_width[0] * get<0>(mortar_coords) + center[0];
    get<1>(mortar_coords) = half_width[1] * get<1>(mortar_coords) + center[1];
    auto neighbor_coords = logical_coordinates(mesh);
    // xi (0) is the perpendicular direction
    get<1>(neighbor_coords) =
        half_width[0] * get<1>(neighbor_coords) + center[0];
    get<2>(neighbor_coords) =
        half_width[1] * get<2>(neighbor_coords) + center[1];

    LocalData self_data{};
    self_data.mortar_data.initialize(mortar_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(self_data.mortar_data) = flux1(face_coords);
    // We need to use the projected values rather than the "exact"
    // values evaluated on the mortar for the method to be
    // conservative.
    self_data.mortar_data = dg::project_to_mortar(
        self_data.mortar_data, mortar_mesh, mortar_mesh, mortar_size);
    self_data.magnitude_of_face_normal = magnitude_of_face_normal1(face_coords);

    LocalData neighbor_data{};
    neighbor_data.mortar_data.initialize(mortar_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(neighbor_data.mortar_data) =
        flux2(mortar_coords);
    neighbor_data.magnitude_of_face_normal =
        magnitude_of_face_normal2(mortar_coords);

    typename mortar_data_tag::type neighbor_mortar_data{};
    neighbor_mortar_data[mortar_id_in_neighbor].local_insert(0, neighbor_data);
    neighbor_mortar_data[mortar_id_in_neighbor].remote_insert(
        0, self_data.mortar_data);

    Variables<tmpl::list<Tags::NormalDotFlux<Var>>> neighbor_flux(
        mortar_mesh.number_of_grid_points());
    get<Tags::NormalDotFlux<Var>>(neighbor_flux) = flux2(mortar_coords);
    typename dt_variables_tag::type neighbor_dt_vars(
        mesh.number_of_grid_points(), 0.);
    add_slice_to_data(
        make_not_null(&neighbor_dt_vars),
        typename dt_variables_tag::type(dg::lift_flux(
            std::move(neighbor_flux), mesh.extents(direction.dimension()),
            magnitude_of_face_normal2(mortar_coords))),
        mesh.extents(), direction.dimension(),
        index_to_slice_at(mesh.extents(), direction));

    tuples::get<MockDistributedObjectsTag>(dist_objects)
        .emplace(neighbor_id, db::create<simple_tags>(
                                  mesh, std::move(neighbor_coords),
                                  typename mortar_meshes_tag::type{
                                      {mortar_id_in_neighbor, mortar_mesh}},
                                  // The neighbors are the small elements, so
                                  // they are the same size as the mortars.
                                  typename mortar_sizes_tag::type{
                                      {mortar_id_in_neighbor,
                                       {{MortarSize::Full, MortarSize::Full}}}},
                                  std::move(neighbor_dt_vars),
                                  std::move(neighbor_mortar_data)));

    auto& self_box = tuples::get<MockDistributedObjectsTag>(dist_objects)
                         .at(self_id)
                         .get_databox<db_type>();
    db::mutate<mortar_data_tag, mortar_meshes_tag, mortar_sizes_tag>(
        make_not_null(&self_box),
        [
          &mortar_id_in_self, &mortar_mesh, &mortar_size, &neighbor_data,
          &self_data
        ](const gsl::not_null<db::item_type<mortar_data_tag>*> data,
          const gsl::not_null<db::item_type<mortar_meshes_tag>*> meshes,
          const gsl::not_null<db::item_type<mortar_sizes_tag>*>
              sizes) noexcept {
          (*data)[mortar_id_in_self].local_insert(0, self_data);
          (*data)[mortar_id_in_self].remote_insert(0,
                                                   neighbor_data.mortar_data);
          (*meshes)[mortar_id_in_self] = mortar_mesh;
          (*sizes)[mortar_id_in_self] = mortar_size;
        });
  };

  add_neighbor({{MortarSize::Full, MortarSize::UpperHalf}}, {{0.0, 0.5}},
               {{1.0, 0.5}}, ElementId<3>{11});
  add_neighbor({{MortarSize::UpperHalf, MortarSize::LowerHalf}}, {{0.5, -0.5}},
               {{0.5, 0.5}}, ElementId<3>{12});
  add_neighbor({{MortarSize::LowerHalf, MortarSize::LowerHalf}}, {{-0.5, -0.5}},
               {{0.5, 0.5}}, ElementId<3>{13});

  MockRuntimeSystem runner{{RefinementNumericalFlux{}},
                           std::move(dist_objects)};
  runner.next_action<my_component>(self_id);

  // These ids don't describe elements that fit together correctly,
  // but they are only used as identifiers so it doesn't matter.
  std::vector<ElementId<3>> ordered_ids{ElementId<3>{11}, ElementId<3>{12},
                                        ElementId<3>{13}};
  for (const auto& id : ordered_ids) {
    runner.next_action<my_component>(id);
  }

  // Check that the operation was conservative.
  const auto get_box = [&runner](const ElementId<3>& id) -> decltype(auto) {
    return runner.algorithms<my_component>().at(id).get_databox<db_type>();
  };
  const auto average_dt = [&mesh](const auto& box,
                                  const auto& det_jacobian) noexcept {
    return definite_integral(
        get(get<Tags::dt<Var>>(box)) *
            get(det_jacobian(get<Tags::Coordinates<3, Frame::Logical>>(box))),
        mesh);
  };
  const double self_average_dt =
      average_dt(get_box(self_id), determinant_of_jacobian1);
  const double neighbors_average_dt =
      0.5 * average_dt(get_box(ordered_ids[0]), determinant_of_jacobian2) +
      0.25 * average_dt(get_box(ordered_ids[1]), determinant_of_jacobian2) +
      0.25 * average_dt(get_box(ordered_ids[2]), determinant_of_jacobian2);
  CHECK(self_average_dt == approx(-neighbors_average_dt));
}
