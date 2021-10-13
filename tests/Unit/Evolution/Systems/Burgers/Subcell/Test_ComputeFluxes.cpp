// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Burgers/Fluxes.hpp"
#include "Evolution/Systems/Burgers/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.ComputeFluxes",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  using argument_tags = typename Fluxes::argument_tags;
  using return_tags = typename Fluxes::return_tags;

  // generate random vars
  const size_t num_pts = 5;
  auto random_vars = make_with_random_values<
      Variables<tmpl::append<return_tags, argument_tags>>>(make_not_null(&gen),
                                                           dist, num_pts);

  // store computed fluxes into return_tags portion of `random_vars`
  subcell::compute_fluxes(make_not_null(&random_vars));

  // compute expected fluxes
  Variables<return_tags> expected_flux{num_pts};
  Fluxes::apply(
      make_not_null(
          &get<::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>(
              expected_flux)),
      get<Tags::U>(random_vars));

  // check result
  CHECK_ITERABLE_APPROX(
      (get<::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>(
          random_vars)),
      (get<::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>(
          expected_flux)));
}
}  // namespace Burgers
