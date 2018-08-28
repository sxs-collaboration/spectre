// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
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
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

class NumericalFlux {
 public:
  using package_tags = tmpl::list<Tags::NormalDotFlux<Var>>;

  void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux_var,
                  const Scalar<DataVector>& ndotf_internal,
                  const Scalar<DataVector>& ndotf_external) const {
    get(*numerical_flux_var) =
        10. * get(ndotf_internal) + 12. * get(ndotf_external);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

struct Metavariables;

using component = ActionTesting::MockArrayComponent<
    Metavariables, domain::ElementIndex<2>,
    tmpl::list<CacheTags::TimeStepper, NumericalFluxTag>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using temporal_id = TimeId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.ApplyBoundaryFluxesLocalTimeStepping",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
  ActionTesting::ActionRunner<Metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(1), NumericalFlux{}}};
  const Slab slab(0., 1.);
  const auto time_step = slab.duration() / 2;
  const auto now = slab.start() + time_step;

  const domain::ElementId<2> id(0);
  const auto face_direction = domain::Direction<2>::upper_xi();
  const auto face_dimension = face_direction.dimension();
  const domain::Mesh<2> mesh(3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto);
  const auto slow_mortar =
      std::make_pair(domain::Direction<2>::upper_xi(),
                     domain::ElementId<2>(1, {{{0, 0}, {1, 0}}}));
  const auto fast_mortar =
      std::make_pair(domain::Direction<2>::upper_xi(),
                     domain::ElementId<2>(1, {{{0, 0}, {1, 1}}}));

  typename Tags::Mortars<domain::Tags::Mesh<1>, 2>::type mortar_meshes{
      {slow_mortar, mesh.slice_away(face_dimension)},
      {fast_mortar, mesh.slice_away(face_dimension)}};
  typename Tags::Mortars<Tags::MortarSize<1>, 2>::type mortar_sizes{
      {slow_mortar,
       dg::mortar_size(id, slow_mortar.second, face_dimension, {})},
      {fast_mortar,
       dg::mortar_size(id, fast_mortar.second, face_dimension, {})}};

  using Vars = Variables<tmpl::list<Var>>;
  Vars variables(mesh.number_of_grid_points(), 2.);

  auto local_data = make_array<2, typename flux_comm_types::LocalData>(
      {typename flux_comm_types::LocalMortarData{
           mesh.slice_away(face_dimension).number_of_grid_points()},
       {}});
  auto remote_data = make_array<3, typename flux_comm_types::PackagedData>(
      mesh.slice_away(face_dimension).number_of_grid_points());
  get(get<Tags::NormalDotFlux<Var>>(gsl::at(local_data, 0).mortar_data)) =
      DataVector{2., 3., 5.};
  get(gsl::at(local_data, 0).magnitude_of_face_normal) =
      DataVector{7., 11., 13.};
  get(get<Tags::NormalDotFlux<Var>>(gsl::at(local_data, 1).mortar_data)) =
      DataVector{17., 19., 23.};
  get(gsl::at(local_data, 1).magnitude_of_face_normal) =
      DataVector{29., 31., 37.};
  get(get<Tags::NormalDotFlux<Var>>(gsl::at(remote_data, 0))) =
      DataVector{41., 43., 47.};
  get(get<Tags::NormalDotFlux<Var>>(gsl::at(remote_data, 1))) =
      DataVector{53., 59., 61.};
  get(get<Tags::NormalDotFlux<Var>>(gsl::at(remote_data, 2))) =
      DataVector{67., 71., 73.};

  using mortar_data_tag =
      typename flux_comm_types::local_time_stepping_mortar_data_tag;
  typename mortar_data_tag::type mortar_data;
  mortar_data[slow_mortar].local_insert(TimeId(true, 0, now),
                                        gsl::at(local_data, 0));
  mortar_data[fast_mortar].local_insert(TimeId(true, 0, now),
                                        gsl::at(local_data, 1));
  mortar_data[slow_mortar].remote_insert(TimeId(true, 0, now - time_step / 2),
                                         gsl::at(remote_data, 0));
  mortar_data[fast_mortar].remote_insert(TimeId(true, 0, now),
                                         gsl::at(remote_data, 1));
  mortar_data[fast_mortar].remote_insert(TimeId(true, 0, now + time_step / 3),
                                         gsl::at(remote_data, 2));

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Mesh<2>, Tags::Mortars<domain::Tags::Mesh<1>, 2>,
      Tags::Mortars<Tags::MortarSize<1>, 2>, Tags::TimeStep,
      System::variables_tag, mortar_data_tag>>(
      mesh, mortar_meshes, mortar_sizes, time_step, variables,
      std::move(mortar_data));

  const auto out_box = get<0>(
      runner
          .apply<component, dg::Actions::ApplyBoundaryFluxesLocalTimeStepping>(
              box, id));

  add_slice_to_data(
      make_not_null(&variables),
      Vars(time_step.value() *
           dg::compute_boundary_flux_contribution<flux_comm_types>(
               NumericalFlux{}, gsl::at(local_data, 0), gsl::at(remote_data, 0),
               mesh.slice_away(face_dimension), mortar_meshes.at(slow_mortar),
               mesh.extents(face_dimension), mortar_sizes.at(slow_mortar))),
      mesh.extents(), face_dimension,
      index_to_slice_at(mesh.extents(), face_direction));

  add_slice_to_data(
      make_not_null(&variables),
      Vars((time_step / 3).value() *
           dg::compute_boundary_flux_contribution<flux_comm_types>(
               NumericalFlux{}, gsl::at(local_data, 1), gsl::at(remote_data, 1),
               mesh.slice_away(face_dimension), mortar_meshes.at(fast_mortar),
               mesh.extents(face_dimension), mortar_sizes.at(fast_mortar))),
      mesh.extents(), face_dimension,
      index_to_slice_at(mesh.extents(), face_direction));

  add_slice_to_data(
      make_not_null(&variables),
      Vars((time_step * 2 / 3).value() *
           dg::compute_boundary_flux_contribution<flux_comm_types>(
               NumericalFlux{}, gsl::at(local_data, 1), gsl::at(remote_data, 2),
               mesh.slice_away(face_dimension), mortar_meshes.at(fast_mortar),
               mesh.extents(face_dimension), mortar_sizes.at(fast_mortar))),
      mesh.extents(), face_dimension,
      index_to_slice_at(mesh.extents(), face_direction));

  CHECK_ITERABLE_APPROX(get<Var>(variables), get<Var>(out_box));
}
