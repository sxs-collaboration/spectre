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
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesGlobalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

/// \cond
// IWYU pragma: no_forward_declare Tensor
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

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

struct Metavariables;

using component = ActionTesting::MockArrayComponent<
    Metavariables, ElementIndex<2>, tmpl::list<NumericalFluxTag>,
    tmpl::list<dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using temporal_id = TemporalId;
  using const_global_cache_tag_list = tmpl::list<>;

  using normal_dot_numerical_flux = NumericalFluxTag;
};

using flux_comm_types = dg::FluxCommunicationTypes<Metavariables>;
using mortar_data_tag = typename flux_comm_types::simple_mortar_data_tag;
using LocalData = typename flux_comm_types::LocalData;
using PackagedData = typename flux_comm_types::PackagedData;
using MagnitudeOfFaceNormal = typename flux_comm_types::MagnitudeOfFaceNormal;
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.ApplyBoundaryFluxesGlobalTimeStepping",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{NumericalFlux{}}};

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const ElementId<2> id(0);

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

    LocalData local_data(3);
    get<Tags::NormalDotFlux<Var>>(local_data) = local_flux;
    get<NumericalFlux::ExtraData>(local_data) = local_extra_data;
    get<Var>(local_data) = local_var;
    get<MagnitudeOfFaceNormal>(local_data) = local_magnitude_face_normal;
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
      Tags::Mesh<2>, Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>,
      mortar_data_tag>>(mesh, initial_dt, std::move(mortar_data));

  const auto out_box = get<0>(
      runner
          .apply<component, dg::Actions::ApplyBoundaryFluxesGlobalTimeStepping>(
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
