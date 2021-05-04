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
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnFdGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
template <typename Tag>
using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

enum class TestThis {
  AllGood,
  SmallDensityDg,
  SmallDensitySubcell,
  SmallPressureDg,
  SmallPressureSubcell,
  PerssonDensity,
  PerssonPressure
};

template <size_t Dim>
void test(const TestThis test_this) {
  CAPTURE(Dim);
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
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using cons_tags = tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>;
  using ConsVars = Variables<cons_tags>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using PrimVars = Variables<prim_tags>;
  using inactive_cons_tags =
      tmpl::list<Inactive<MassDensityCons>, Inactive<MomentumDensity>,
                 Inactive<EnergyDensity>>;
  using InactiveConsVars = Variables<inactive_cons_tags>;
  using inactive_prim_tags =
      tmpl::list<Inactive<MassDensity>, Inactive<Velocity>,
                 Inactive<SpecificInternalEnergy>, Inactive<Pressure>>;
  using InactivePrimVars = Variables<inactive_prim_tags>;

  const double persson_exponent = 4.0;
  std::unique_ptr<EquationsOfState::EquationOfState<false, 2>> eos =
      std::make_unique<EquationsOfState::IdealFluid<false>>(5.0 / 3.0);
  ConsVars subcell_cons{subcell_mesh.number_of_grid_points()};
  PrimVars subcell_prims{subcell_mesh.number_of_grid_points(), 1.0e-7};
  InactivePrimVars dg_prim{dg_mesh.number_of_grid_points(), 1.0e-7};
  InactiveConsVars dg_cons{dg_mesh.number_of_grid_points()};

  if (test_this == TestThis::SmallPressureDg) {
    get(get<Inactive<Pressure>>(dg_prim))[dg_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::PerssonPressure) {
    get(get<Inactive<Pressure>>(dg_prim))[dg_mesh.number_of_grid_points() / 2] =
        1.0;
  }

  get<Inactive<SpecificInternalEnergy>>(dg_prim) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<Inactive<MassDensity>>(dg_prim),
          get<Inactive<Pressure>>(dg_prim));
  NewtonianEuler::ConservativeFromPrimitive<Dim>::apply(
      make_not_null(&get<Inactive<MassDensityCons>>(dg_cons)),
      make_not_null(&get<Inactive<MomentumDensity>>(dg_cons)),
      make_not_null(&get<Inactive<EnergyDensity>>(dg_cons)),
      get<Inactive<MassDensity>>(dg_prim), get<Inactive<Velocity>>(dg_prim),
      get<Inactive<SpecificInternalEnergy>>(dg_prim));

  if (test_this == TestThis::SmallDensitySubcell) {
    get(get<MassDensity>(
        subcell_prims))[subcell_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::SmallPressureSubcell) {
    get(get<Pressure>(
        subcell_prims))[subcell_mesh.number_of_grid_points() / 2] =
        0.1 * 1.0e-18;
  } else if (test_this == TestThis::SmallDensityDg) {
    get(get<Inactive<MassDensityCons>>(
        dg_cons))[dg_mesh.number_of_grid_points() / 2] = 0.1 * 1.0e-18;
  } else if (test_this == TestThis::PerssonDensity) {
    get(get<Inactive<MassDensityCons>>(
        dg_cons))[dg_mesh.number_of_grid_points() / 2] = 1.0e18;
  }

  get<SpecificInternalEnergy>(subcell_prims) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<MassDensity>(subcell_prims), get<Pressure>(subcell_prims));

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<cons_tags>, ::Tags::Variables<prim_tags>,
      Inactive<::Tags::Variables<cons_tags>>, ::domain::Tags::Mesh<Dim>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<false, 2>>>>>(
      subcell_cons, subcell_prims, dg_cons, dg_mesh, std::move(eos));
  db::mutate_apply<NewtonianEuler::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));
  const bool result =
      db::mutate_apply<NewtonianEuler::subcell::TciOnFdGrid<Dim>>(
          make_not_null(&box), persson_exponent);
  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(result);
  } else {
    CHECK(result);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Subcell.TciOnFdGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this :
       {TestThis::AllGood, TestThis::SmallDensityDg,
        TestThis::SmallDensitySubcell, TestThis::SmallPressureDg,
        TestThis::SmallPressureSubcell, TestThis::PerssonDensity,
        TestThis::PerssonPressure}) {
    test<1>(test_this);
    test<2>(test_this);
    test<3>(test_this);
  }
}
