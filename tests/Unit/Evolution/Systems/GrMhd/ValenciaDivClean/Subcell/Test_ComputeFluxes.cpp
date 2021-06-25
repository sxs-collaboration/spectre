// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.ComputeFluxes",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const size_t num_pts = 5;

  auto vars = make_with_random_values<Variables<
      tmpl::append<ComputeFluxes::return_tags, ComputeFluxes::argument_tags>>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);

  Variables<ComputeFluxes::return_tags> expected_fluxes{num_pts};
  ComputeFluxes::apply(
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<>,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<>,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeS<>>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(vars),
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(vars),
      get<gr::Tags::Lapse<>>(vars), get<gr::Tags::Shift<3>>(vars),
      get<gr::Tags::SqrtDetSpatialMetric<>>(vars),
      get<gr::Tags::SpatialMetric<3>>(vars),
      get<gr::Tags::InverseSpatialMetric<3>>(vars),
      get<hydro::Tags::Pressure<DataVector>>(vars),
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars),
      get<hydro::Tags::LorentzFactor<DataVector>>(vars),
      get<hydro::Tags::MagneticField<DataVector, 3>>(vars));

  subcell::compute_fluxes(make_not_null(&vars));

  tmpl::for_each<ComputeFluxes::return_tags>(
      [&expected_fluxes, &vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(expected_fluxes));
      });
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
