// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ComputeFluxesFromPrimitives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.ComputeFluxesFromPrimitives",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const size_t num_pts = 5;

  using ConservativeTags =
      typename grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags;
  using ArgumentTags = typename grmhd::ValenciaDivClean::
      ConservativeFromPrimitive::argument_tags;

  auto flux_vars =
      make_with_random_values<Variables<ComputeFluxes::return_tags>>(
          make_not_null(&gen), make_not_null(&dist), num_pts);

  auto conservative_vars = make_with_random_values<Variables<ConservativeTags>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);

  auto boundary_vars = make_with_random_values<Variables<tmpl::push_back<
      ArgumentTags, gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>>>>(
      make_not_null(&gen), make_not_null(&dist), num_pts);

  Variables<ConservativeFromPrimitive::return_tags> expected_conservative{
      num_pts};
  Variables<ComputeFluxes::return_tags> expected_fluxes{num_pts};
  ConservativeFromPrimitive::apply(
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildeD>(expected_conservative)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildeYe>(expected_conservative)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildeTau>(expected_conservative)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildeS<>>(expected_conservative)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildeB<>>(expected_conservative)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::TildePhi>(expected_conservative)),
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_vars),
      get<hydro::Tags::ElectronFraction<DataVector>>(boundary_vars),
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_vars),
      get<hydro::Tags::Pressure<DataVector>>(boundary_vars),
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_vars),
      get<hydro::Tags::LorentzFactor<DataVector>>(boundary_vars),
      get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_vars),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_vars),
      get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_vars),
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(boundary_vars));
  ComputeFluxes::apply(
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD,
                                      tmpl::size_t<3>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe,
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
      get<grmhd::ValenciaDivClean::Tags::TildeD>(expected_conservative),
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(expected_conservative),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(expected_conservative),
      get<grmhd::ValenciaDivClean::Tags::TildeS<>>(expected_conservative),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(expected_conservative),
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(expected_conservative),
      get<gr::Tags::Lapse<DataVector>>(boundary_vars),
      get<gr::Tags::Shift<DataVector, 3>>(boundary_vars),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(boundary_vars),
      get<gr::Tags::SpatialMetric<DataVector, 3>>(boundary_vars),
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(boundary_vars),
      get<hydro::Tags::Pressure<DataVector>>(boundary_vars),
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(boundary_vars),
      get<hydro::Tags::LorentzFactor<DataVector>>(boundary_vars),
      get<hydro::Tags::MagneticField<DataVector, 3>>(boundary_vars));

  compute_fluxes_from_primitives(make_not_null(&flux_vars), boundary_vars);

  tmpl::for_each<ComputeFluxes::return_tags>(
      [&expected_fluxes, &flux_vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(flux_vars), get<tag>(expected_fluxes));
      });
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
