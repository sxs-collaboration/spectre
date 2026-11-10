// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianMhd/Fluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/Subcell/ComputeFluxes.hpp"
#include "Evolution/Systems/NewtonianMhd/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  const size_t num_pts = 5;

  auto vars = make_with_random_values<Variables<
      tmpl::append<typename NewtonianMhd::ComputeFluxes<Dim>::return_tags,
                   typename NewtonianMhd::ComputeFluxes<Dim>::argument_tags>>>(
      gen, dist, num_pts);

  Variables<typename NewtonianMhd::ComputeFluxes<Dim>::return_tags>
      expected_fluxes{num_pts};
  NewtonianMhd::ComputeFluxes<Dim>::apply(
      make_not_null(&get<::Tags::Flux<NewtonianMhd::Tags::MassDensityCons,
                                      tmpl::size_t<Dim>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<NewtonianMhd::Tags::MomentumDensity<Dim>,
                                      tmpl::size_t<Dim>, Frame::Inertial>>(
          expected_fluxes)),
      make_not_null(&get<::Tags::Flux<NewtonianMhd::Tags::EnergyDensity,
                                      tmpl::size_t<Dim>, Frame::Inertial>>(
          expected_fluxes)),
      get<NewtonianMhd::Tags::MomentumDensity<Dim>>(vars),
      get<NewtonianMhd::Tags::EnergyDensity>(vars),
      get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(vars),
      get<hydro::Tags::Pressure<DataVector>>(vars));

  NewtonianMhd::subcell::compute_fluxes<Dim>(make_not_null(&vars));

  tmpl::for_each<typename NewtonianMhd::ComputeFluxes<Dim>::return_tags>(
      [&expected_fluxes, &vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        const auto& computed_flux = get<tag>(vars);
        const auto& expected_flux = get<tag>(expected_fluxes);
        CHECK_ITERABLE_APPROX(computed_flux, expected_flux);
      });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianMhd.Subcell.ComputeFluxes",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  test<1>(make_not_null(&gen), make_not_null(&dist));
  test<2>(make_not_null(&gen), make_not_null(&dist));
  test<3>(make_not_null(&gen), make_not_null(&dist));
}
