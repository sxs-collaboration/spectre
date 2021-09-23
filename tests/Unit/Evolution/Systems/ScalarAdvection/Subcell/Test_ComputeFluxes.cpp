// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  using argument_tags = typename Fluxes<Dim>::argument_tags;
  using return_tags = typename Fluxes<Dim>::return_tags;

  // generate random vars
  const size_t num_pts = 5;
  auto random_vars = make_with_random_values<
      Variables<tmpl::append<return_tags, argument_tags>>>(gen, dist, num_pts);

  // store computed fluxes into return_tags portion of `random_vars`
  subcell::compute_fluxes<Dim>(make_not_null(&random_vars));

  // compute expected fluxes
  Variables<return_tags> expected_flux{num_pts};
  Fluxes<Dim>::apply(
      make_not_null(
          &get<::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>(
              expected_flux)),
      get<Tags::U>(random_vars), get<Tags::VelocityField<Dim>>(random_vars));

  // check result
  CHECK_ITERABLE_APPROX(
      (get<::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>(
          random_vars)),
      (get<::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>(
          expected_flux)));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarAdvection.Subcell.ComputeFluxes",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  test<1>(make_not_null(&gen), make_not_null(&dist));
  test<2>(make_not_null(&gen), make_not_null(&dist));
}
}  // namespace ScalarAdvection
