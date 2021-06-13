// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist,
          const bool did_rollback) {
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;

  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using ConsVars = Variables<cons_tags>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using PrimVars = Variables<prim_tags>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  auto cons_vars = make_with_random_values<ConsVars>(
      gen, dist, subcell_mesh.number_of_grid_points());
  PrimVars expected_prim_vars{};

  std::unique_ptr<EquationsOfState::EquationOfState<false, 1>> eos =
      std::make_unique<EquationsOfState::PolytropicFluid<false>>(1.4,
                                                                 5.0 / 3.0);

  auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::DidRollback, ::Tags::Variables<cons_tags>,
      ::Tags::Variables<prim_tags>, ::domain::Tags::Mesh<Dim>,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<false, 1>>>>>(
      did_rollback, cons_vars, expected_prim_vars, dg_mesh, subcell_mesh,
      std::move(eos));

  db::mutate_apply<NewtonianEuler::subcell::PrimsAfterRollback<Dim>>(
      make_not_null(&box));

  if (did_rollback) {
    REQUIRE(
        db::get<::Tags::Variables<prim_tags>>(box).number_of_grid_points() ==
        cons_vars.number_of_grid_points());
    expected_prim_vars.initialize(cons_vars.number_of_grid_points());
    NewtonianEuler::PrimitiveFromConservative<Dim, 1>::apply(
        make_not_null(&get<MassDensity>(expected_prim_vars)),
        make_not_null(&get<Velocity>(expected_prim_vars)),
        make_not_null(&get<SpecificInternalEnergy>(expected_prim_vars)),
        make_not_null(&get<Pressure>(expected_prim_vars)),
        get<MassDensityCons>(cons_vars), get<MomentumDensity>(cons_vars),
        get<EnergyDensity>(cons_vars),
        db::get<hydro::Tags::EquationOfStateBase>(box));
    CHECK_VARIABLES_APPROX(db::get<::Tags::Variables<prim_tags>>(box),
                           expected_prim_vars);
  } else {
    CHECK(db::get<::Tags::Variables<prim_tags>>(box).number_of_grid_points() ==
          0);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.PrimsAfterRollback",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 1.0);
  for (const bool did_rollback : {true, false}) {
    test<1>(make_not_null(&gen), make_not_null(&dist), did_rollback);
    test<2>(make_not_null(&gen), make_not_null(&dist), did_rollback);
    test<3>(make_not_null(&gen), make_not_null(&dist), did_rollback);
  }
}
