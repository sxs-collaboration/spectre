// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnDgGrid.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTci.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
template <size_t ThermodynamicDim>
std::tuple<bool, evolution::dg::subcell::RdmpTciData> TciOnDgGrid<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        dg_prim_vars,
    const Variables<
        tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>& dg_vars,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
    const evolution::dg::subcell::SubcellOptions& subcell_options,
    const double persson_exponent,
    [[maybe_unused]] const bool element_stays_on_dg) {
  const Variables<tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>
      subcell_vars = evolution::dg::subcell::fd::project(
          dg_vars, dg_mesh, subcell_mesh.extents());
  const Scalar<DataVector>& mass_density = get<MassDensityCons>(dg_vars);
  const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density =
      get<MomentumDensity>(dg_vars);
  const Scalar<DataVector>& energy_density = get<EnergyDensity>(dg_vars);

  const Scalar<DataVector>& subcell_mass_density =
      get<MassDensityCons>(subcell_vars);
  const Scalar<DataVector>& subcell_energy_density =
      get<EnergyDensity>(subcell_vars);

  using std::max;
  using std::min;
  evolution::dg::subcell::RdmpTciData rdmp_tci_data{
      {max(max(get(mass_density)), max(get(subcell_mass_density))),
       max(max(get(energy_density)), max(get(subcell_energy_density)))},
      {min(min(get(mass_density)), min(get(subcell_mass_density))),
       min(min(get(energy_density)), min(get(subcell_energy_density)))}};

  NewtonianEuler::PrimitiveFromConservative<Dim>::apply(
      make_not_null(&get<MassDensity>(*dg_prim_vars)),
      make_not_null(&get<Velocity>(*dg_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(*dg_prim_vars)),
      make_not_null(&get<Pressure>(*dg_prim_vars)), mass_density,
      momentum_density, energy_density, eos);

  const bool cell_is_troubled =
      evolution::dg::subcell::rdmp_tci(rdmp_tci_data.max_variables_values,
                                       rdmp_tci_data.min_variables_values,
                                       past_rdmp_tci_data.max_variables_values,
                                       past_rdmp_tci_data.min_variables_values,
                                       subcell_options.rdmp_delta0(),
                                       subcell_options.rdmp_epsilon()) or
      min(get(mass_density)) < min_density_allowed or
      min(get(get<Pressure>(*dg_prim_vars))) < min_pressure_allowed or
      evolution::dg::subcell::persson_tci(mass_density, dg_mesh,
                                          persson_exponent) or
      evolution::dg::subcell::persson_tci(energy_density, dg_mesh,
                                          persson_exponent);
  return {cell_is_troubled, std::move(rdmp_tci_data)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class TciOnDgGrid<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template std::tuple<bool, evolution::dg::subcell::RdmpTciData>              \
  TciOnDgGrid<DIM(data)>::apply<THERMO_DIM(data)>(                            \
      gsl::not_null<Variables<tmpl::list<MassDensity, Velocity,               \
                                         SpecificInternalEnergy, Pressure>>*> \
          dg_prim_vars,                                                       \
      const Variables<                                                        \
          tmpl::list<NewtonianEuler::Tags::MassDensityCons,                   \
                     NewtonianEuler::Tags::MomentumDensity<DIM(data)>,        \
                     NewtonianEuler::Tags::EnergyDensity>>& dg_vars,          \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,  \
      const Mesh<DIM(data)>& dg_mesh, const Mesh<DIM(data)>& subcell_mesh,    \
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,          \
      const evolution::dg::subcell::SubcellOptions& subcell_options,          \
      double persson_exponent, const bool element_stays_on_dg);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::subcell
