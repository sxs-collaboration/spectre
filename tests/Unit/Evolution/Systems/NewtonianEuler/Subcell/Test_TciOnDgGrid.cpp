// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnDgGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
enum class TestThis {
  AllGood,
  SmallDensity,
  SmallPressure,
  PerssonDensity,
  PerssonPressure
};

template <size_t Dim>
void test(const TestThis test_this) {
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;

  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using ConsVars = Variables<cons_tags>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using PrimVars = Variables<prim_tags>;

  const double persson_exponent = 4.0;
  PrimVars dg_prims{dg_mesh.number_of_grid_points(), 1.0e-7};
  std::unique_ptr<EquationsOfState::EquationOfState<false, 2>> eos =
      std::make_unique<EquationsOfState::IdealFluid<false>>(5.0 / 3.0);
  if (test_this == TestThis::SmallDensity) {
    get(get<MassDensity>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::SmallPressure) {
    get(get<Pressure>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::PerssonDensity) {
    get(get<MassDensity>(dg_prims))[dg_mesh.number_of_grid_points() / 2] =
        1.0e18;
  } else if (test_this == TestThis::PerssonPressure) {
    get(get<Pressure>(dg_prims))[dg_mesh.number_of_grid_points() / 2] = 1.0e18;
  }

  get<SpecificInternalEnergy>(dg_prims) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<MassDensity>(dg_prims), get<Pressure>(dg_prims));

  auto box = db::create<
      db::AddSimpleTags<::Tags::Variables<cons_tags>,
                        ::Tags::Variables<prim_tags>, ::domain::Tags::Mesh<Dim>,
                        hydro::Tags::EquationOfState<std::unique_ptr<
                            EquationsOfState::EquationOfState<false, 2>>>>>(
      ConsVars{dg_mesh.number_of_grid_points()}, dg_prims, dg_mesh,
      std::move(eos));
  db::mutate_apply<NewtonianEuler::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));
  const bool result =
      db::mutate_apply<NewtonianEuler::subcell::TciOnDgGrid<Dim>>(
          make_not_null(&box), persson_exponent);
  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(result);
  } else {
    CHECK(result);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this :
       {TestThis::AllGood, TestThis::SmallDensity, TestThis::SmallPressure,
        TestThis::PerssonDensity, TestThis::PerssonPressure}) {
    test<1>(test_this);
    test<2>(test_this);
    test<3>(test_this);
  }
}
