// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
// IWYU pragma: no_include <unordered_map>
#include <utility>

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
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
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

template <size_t Dim, typename Flux>
struct Metavariables;

template <size_t Dim, typename Flux>
using component = ActionTesting::MockArrayComponent<
    Metavariables<Dim, Flux>, ElementIndex<Dim>,
    tmpl::list<NumericalFluxTag<Flux>>,
    tmpl::list<dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>>;

template <size_t Dim, typename Flux>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<component<Dim, Flux>>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag<Flux>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.ApplyBoundaryFluxesGlobalTimeStepping",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using flux_comm_types =
      dg::FluxCommunicationTypes<Metavariables<2, NumericalFlux>>;
  using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
  using LocalData = typename flux_comm_types::LocalData;
  using PackagedData = typename flux_comm_types::PackagedData;
  using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<1>, 2>;

  ActionTesting::ActionRunner<Metavariables<2, NumericalFlux>> runner{
      {NumericalFlux{}}};

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const ElementId<2> id(0);

  typename mortar_meshes_tag::type mortar_meshes{
      {{Direction<2>::upper_xi(), id}, mesh.slice_away(0)},
      {{Direction<2>::lower_eta(), id}, mesh.slice_away(1)}};

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

  auto box = db::create<db::AddSimpleTags<
      Tags::Mesh<2>, mortar_meshes_tag,
      Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>, mortar_data_tag>>(
      mesh, std::move(mortar_meshes), initial_dt, std::move(mortar_data));

  const auto out_box =
      get<0>(runner.apply<component<2, NumericalFlux>,
                          dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>(
          box, id));

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
      get(db::get<Tags::dt<Var>>(out_box)),
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
    "Unit.DG.Actions.ApplyBoundaryFluxesGlobalTimeStepping.p-refinement",
    "[Unit][NumericalAlgorithms][Actions]") {
  using flux_comm_types =
      dg::FluxCommunicationTypes<Metavariables<3, RefinementNumericalFlux>>;
  using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
  using LocalData = typename flux_comm_types::LocalData;
  using PackagedData = typename flux_comm_types::PackagedData;
  using mortar_meshes_tag = Tags::Mortars<Tags::Mesh<2>, 3>;
  using dt_variables_tag =
      db::add_tag_prefix<Tags::dt, typename System<3>::variables_tag>;

  ActionTesting::ActionRunner<Metavariables<3, RefinementNumericalFlux>> runner{
      {RefinementNumericalFlux{}}};

  const ElementId<3> id(0);
  const auto direction = Direction<3>::upper_xi();

  const auto make_initial_box = [](
      const std::array<size_t, 3>& extents) noexcept {
    const Mesh<3> mesh(extents, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto);
    return db::create<db::AddSimpleTags<Tags::Mesh<3>, mortar_meshes_tag,
                                        dt_variables_tag, mortar_data_tag>>(
        mesh, typename mortar_meshes_tag::type{},
        typename dt_variables_tag::type(mesh.number_of_grid_points(), 0.),
        typename mortar_data_tag::type{});
  };
  auto box1 = make_initial_box({{3, 4, 5}});
  auto box2 = make_initial_box({{4, 2, 6}});

  const auto set_boundary_data =
      [&direction, &id](const auto local, const auto remote, const auto flux,
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
        dg::project_to_mortar(local_data.mortar_data, face_mesh, mortar_mesh);

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

    db::mutate<dt_variables_tag, mortar_meshes_tag, mortar_data_tag>(
        local,
        [
          &direction, &id, &local_data, &mortar_mesh, &uncoupled_lifted_flux,
          &volume_mesh
        ](const gsl::not_null<db::item_type<dt_variables_tag>*> dt_variables,
          const gsl::not_null<db::item_type<mortar_meshes_tag>*> mortar_meshes,
          const gsl::not_null<db::item_type<mortar_data_tag>*>
              mortar_data) noexcept {
          const auto mortar_id = std::make_pair(direction, id);
          add_slice_to_data(
              dt_variables, uncoupled_lifted_flux, volume_mesh.extents(),
              direction.dimension(),
              index_to_slice_at(volume_mesh.extents(), direction));
          (*mortar_meshes)[mortar_id] = mortar_mesh;
          (*mortar_data)[mortar_id].local_insert(0, std::move(local_data));
        });
    db::mutate<mortar_data_tag>(
        remote, [&direction, &id, &remote_data](
                    const gsl::not_null<db::item_type<mortar_data_tag>*>
                        mortar_data) noexcept {
          const auto mortar_id = std::make_pair(direction, id);
          (*mortar_data)[mortar_id].remote_insert(0, std::move(remote_data));
        });
  };
  set_boundary_data(make_not_null(&box1), make_not_null(&box2), flux1,
                    magnitude_of_face_normal1);
  set_boundary_data(make_not_null(&box2), make_not_null(&box1), flux2,
                    magnitude_of_face_normal2);

  const auto out_box1 =
      get<0>(runner.apply<component<3, RefinementNumericalFlux>,
                          dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>(
          box1, id));
  const auto out_box2 =
      get<0>(runner.apply<component<3, RefinementNumericalFlux>,
                          dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>(
          box2, id));

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
