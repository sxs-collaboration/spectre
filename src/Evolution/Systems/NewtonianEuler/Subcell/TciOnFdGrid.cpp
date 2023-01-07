// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnFdGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
template <size_t ThermodynamicDim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnFdGrid<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        subcell_grid_prim_vars,
    const Variables<tmpl::list<MassDensityCons, MomentumDensity,
                               EnergyDensity>>& subcell_vars,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const double persson_exponent, const bool need_rdmp_data_only) {
  const Scalar<DataVector>& subcell_mass_density =
      get<MassDensityCons>(subcell_vars);
  const tnsr::I<DataVector, Dim, Frame::Inertial>& subcell_momentum_density =
      get<MomentumDensity>(subcell_vars);
  const Scalar<DataVector>& subcell_energy_density =
      get<EnergyDensity>(subcell_vars);
  const auto dg_vars = evolution::dg::subcell::fd::reconstruct(
      subcell_vars, dg_mesh, subcell_mesh.extents(),
      evolution::dg::subcell::fd::ReconstructionMethod::DimByDim);
  const Scalar<DataVector>& dg_mass_density = get<MassDensityCons>(dg_vars);
  const tnsr::I<DataVector, Dim, Frame::Inertial>& dg_momentum_density =
      get<MomentumDensity>(dg_vars);
  const Scalar<DataVector>& dg_energy_density = get<EnergyDensity>(dg_vars);

  NewtonianEuler::PrimitiveFromConservative<Dim>::apply(
      make_not_null(&get<MassDensity>(*subcell_grid_prim_vars)),
      make_not_null(&get<Velocity>(*subcell_grid_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(*subcell_grid_prim_vars)),
      make_not_null(&get<Pressure>(*subcell_grid_prim_vars)),
      subcell_mass_density, subcell_momentum_density, subcell_energy_density,
      eos);
  Variables<tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>
      dg_grid_prim_vars{get(dg_energy_density).size()};
  NewtonianEuler::PrimitiveFromConservative<Dim>::apply(
      make_not_null(&get<MassDensity>(dg_grid_prim_vars)),
      make_not_null(&get<Velocity>(dg_grid_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(dg_grid_prim_vars)),
      make_not_null(&get<Pressure>(dg_grid_prim_vars)), dg_mass_density,
      dg_momentum_density, dg_energy_density, eos);

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(get(subcell_mass_density)), max(get(subcell_energy_density))},
      {min(get(subcell_mass_density)), min(get(subcell_energy_density))}};

  const evolution::dg::subcell::RdmpTciData rdmp_tci_data_for_check{
      {max(rdmp_tci_data.max_variables_values[0], max(get(dg_mass_density))),
       max(rdmp_tci_data.max_variables_values[1], max(get(dg_energy_density)))},
      {min(rdmp_tci_data.min_variables_values[0], min(get(dg_mass_density))),
       min(rdmp_tci_data.min_variables_values[1],
           min(get(dg_energy_density)))}};

  if (need_rdmp_data_only) {
    return {false, rdmp_tci_data};
  }

  const bool cell_is_troubled =
      evolution::dg::subcell::rdmp_tci(
          rdmp_tci_data_for_check.max_variables_values,
          rdmp_tci_data_for_check.min_variables_values,
          past_rdmp_tci_data.max_variables_values,
          past_rdmp_tci_data.min_variables_values,
          subcell_options.rdmp_delta0(), subcell_options.rdmp_epsilon()) or
      min(min(get(subcell_mass_density)), min(get(dg_mass_density))) <
          min_density_allowed or
      min(min(get(get<Pressure>(*subcell_grid_prim_vars))),
          min(get(get<Pressure>(dg_grid_prim_vars)))) < min_pressure_allowed or
      evolution::dg::subcell::persson_tci(dg_mass_density, dg_mesh,
                                          persson_exponent) or
      evolution::dg::subcell::persson_tci(dg_energy_density, dg_mesh,
                                          persson_exponent);

  return {cell_is_troubled, std::move(rdmp_tci_data)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class TciOnFdGrid<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template std::tuple<bool, evolution::dg::subcell::RdmpTciData>              \
  TciOnFdGrid<DIM(data)>::apply<THERMO_DIM(data)>(                            \
      gsl::not_null<Variables<tmpl::list<MassDensity, Velocity,               \
                                         SpecificInternalEnergy, Pressure>>*> \
          subcell_grid_prim_vars,                                             \
      const Variables<                                                        \
          tmpl::list<NewtonianEuler::Tags::MassDensityCons,                   \
                     NewtonianEuler::Tags::MomentumDensity<DIM(data)>,        \
                     NewtonianEuler::Tags::EnergyDensity>>& subcell_vars,     \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,  \
      const Mesh<DIM(data)>& dg_mesh, const Mesh<DIM(data)>& subcell_mesh,    \
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,          \
      const evolution::dg::subcell::SubcellOptions& subcell_options,          \
      double persson_exponent, bool need_rdmp_data_only);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::subcell
