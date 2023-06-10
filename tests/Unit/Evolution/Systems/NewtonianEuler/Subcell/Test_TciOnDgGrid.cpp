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
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/SubcellOptions.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
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
  PerssonEnergyDensity,
  RdmpMassDensity,
  RdmpEnergyDensity
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
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

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
  } else if (test_this == TestThis::PerssonEnergyDensity) {
    get(get<Pressure>(dg_prims))[dg_mesh.number_of_grid_points() / 2] = 1.0e18;
  }

  get<SpecificInternalEnergy>(dg_prims) =
      eos->specific_internal_energy_from_density_and_pressure(
          get<MassDensity>(dg_prims), get<Pressure>(dg_prims));

  const evolution::dg::subcell::SubcellOptions subcell_options{
      1.0e-18,
      1.0e-4,
      1.0e-18,
      1.0e-4,
      persson_exponent,
      persson_exponent,
      false,
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim,
      false,
      std::nullopt,
      fd::DerivativeOrder::Two};

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<cons_tags>, ::Tags::Variables<prim_tags>,
      ::domain::Tags::Mesh<Dim>, ::evolution::dg::subcell::Tags::Mesh<Dim>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<false, 2>>>,
      evolution::dg::subcell::Tags::SubcellOptions<Dim>,
      evolution::dg::subcell::Tags::DataForRdmpTci>>(
      ConsVars{dg_mesh.number_of_grid_points()}, dg_prims, dg_mesh,
      subcell_mesh, std::move(eos), subcell_options,
      evolution::dg::subcell::RdmpTciData{});
  db::mutate_apply<NewtonianEuler::ConservativeFromPrimitive<Dim>>(
      make_not_null(&box));

  // Set the RDMP TCI past data.
  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData past_rdmp_tci_data{
      {max(max(get(get<MassDensityCons>(box))),
           max(evolution::dg::subcell::fd::project(
               get(get<MassDensityCons>(box)), dg_mesh,
               subcell_mesh.extents()))),
       max(max(get(get<EnergyDensity>(box))),
           max(evolution::dg::subcell::fd::project(get(get<EnergyDensity>(box)),
                                                   dg_mesh,
                                                   subcell_mesh.extents())))},
      {min(min(get(get<MassDensityCons>(box))),
           min(evolution::dg::subcell::fd::project(
               get(get<MassDensityCons>(box)), dg_mesh,
               subcell_mesh.extents()))),
       min(min(get(get<EnergyDensity>(box))),
           min(evolution::dg::subcell::fd::project(get(get<EnergyDensity>(box)),
                                                   dg_mesh,
                                                   subcell_mesh.extents())))}};

  const evolution::dg::subcell::RdmpTciData expected_rdmp_tci_data =
      past_rdmp_tci_data;

  // Modify past data if we are expected an RDMP TCI failure.
  db::mutate<evolution::dg::subcell::Tags::DataForRdmpTci>(
      [&past_rdmp_tci_data, &test_this](const auto rdmp_tci_data_ptr) {
        *rdmp_tci_data_ptr = past_rdmp_tci_data;
        if (test_this == TestThis::RdmpMassDensity) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[0] *= 1.01;
        } else if (test_this == TestThis::RdmpEnergyDensity) {
          // Assumes min is positive, increase it so we fail the TCI
          rdmp_tci_data_ptr->min_variables_values[1] *= 1.01;
        }
      },
      make_not_null(&box));

  const bool element_stays_on_dg = false;
  const std::tuple<bool, evolution::dg::subcell::RdmpTciData> result =
      db::mutate_apply<NewtonianEuler::subcell::TciOnDgGrid<Dim>>(
          make_not_null(&box), persson_exponent, element_stays_on_dg);

  CHECK_ITERABLE_APPROX(get<1>(result).max_variables_values,
                        expected_rdmp_tci_data.max_variables_values);
  CHECK_ITERABLE_APPROX(get<1>(result).min_variables_values,
                        expected_rdmp_tci_data.min_variables_values);
  if (test_this == TestThis::AllGood) {
    CHECK_FALSE(std::get<0>(result));
  } else {
    CHECK(std::get<0>(result));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const auto test_this :
       {TestThis::AllGood, TestThis::SmallDensity, TestThis::SmallPressure,
        TestThis::PerssonDensity, TestThis::PerssonEnergyDensity,
        TestThis::RdmpMassDensity, TestThis::RdmpEnergyDensity}) {
    test<1>(test_this);
    test<2>(test_this);
    test<3>(test_this);
  }
}
