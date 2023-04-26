// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ForceFree/Fluxes.hpp"
#include "Evolution/Systems/ForceFree/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
namespace {
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  using argument_tags = typename Fluxes::argument_tags;
  using return_tags = typename Fluxes::return_tags;

  const size_t num_pts = 5;
  auto random_vars = make_with_random_values<
      Variables<tmpl::append<return_tags, argument_tags>>>(gen, dist, num_pts);

  // store computed fluxes into return_tags portion of `random_vars`
  subcell::compute_fluxes(make_not_null(&random_vars));

  // compute expected fluxes
  Variables<return_tags> expected_fluxes{num_pts};
  Fluxes::apply(
      make_not_null(
          &get<::Tags::Flux<Tags::TildeE, tmpl::size_t<3>, Frame::Inertial>>(
              expected_fluxes)),
      make_not_null(
          &get<::Tags::Flux<Tags::TildeB, tmpl::size_t<3>, Frame::Inertial>>(
              expected_fluxes)),
      make_not_null(
          &get<::Tags::Flux<Tags::TildePsi, tmpl::size_t<3>, Frame::Inertial>>(
              expected_fluxes)),
      make_not_null(
          &get<::Tags::Flux<Tags::TildePhi, tmpl::size_t<3>, Frame::Inertial>>(
              expected_fluxes)),
      make_not_null(
          &get<::Tags::Flux<Tags::TildeQ, tmpl::size_t<3>, Frame::Inertial>>(
              expected_fluxes)),
      get<Tags::TildeE>(random_vars), get<Tags::TildeB>(random_vars),
      get<Tags::TildePsi>(random_vars), get<Tags::TildePhi>(random_vars),
      get<Tags::TildeQ>(random_vars), get<Tags::TildeJ>(random_vars),
      get<gr::Tags::Lapse<DataVector>>(random_vars),
      get<gr::Tags::Shift<DataVector, 3>>(random_vars),
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(random_vars),
      get<gr::Tags::SpatialMetric<DataVector, 3>>(random_vars),
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(random_vars));

  tmpl::for_each<Fluxes::return_tags>(
      [&expected_fluxes, &random_vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(random_vars), get<tag>(expected_fluxes));
      });
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.ComputeFluxes",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  test(make_not_null(&gen), make_not_null(&dist));
}

}  // namespace
}  // namespace ForceFree
