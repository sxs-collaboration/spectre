// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/SubcellOptions.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace NewtonianEuler::subcell {
/*!
 * \brief Troubled-cell indicator applied to the finite difference subcell
 * solution to check if the corresponding DG solution is admissible.
 *
 * Computes the primitive variables on the DG and subcell grids, mutating the
 * subcell/active primitive variables in the DataBox. Then,
 * - apply RDMP TCI to the mass and energy density
 * - if the minimum density or pressure on either the DG or subcell mesh are
 *   below \f$10^{-18}\f$, marks the element as troubled and returns. We check
 *   both the FD and DG grids since when a discontinuity is inside the element
 *   oscillations in the DG solution can result in negative values that aren't
 *   present in the FD solution.
 * - runs the Persson TCI on the mass and energy density on the DG grid. The
 *   reason for applying the Persson TCI to both the mass and energy density is
 *   to flag cells at contact discontinuities. The Persson TCI only works with
 *   spectral-type methods and is a direct check of whether or not the DG
 *   solution is a good representation of the underlying data.
 *
 * Please note that the TCI is run after the subcell solution has been
 * reconstructed to the DG grid, and so `Inactive<Tag>` is the updated DG
 * solution.
 */
template <size_t Dim>
class TciOnFdGrid {
 private:
  using MassDensityCons = NewtonianEuler::Tags::MassDensityCons;
  using EnergyDensity = NewtonianEuler::Tags::EnergyDensity;
  using MomentumDensity = NewtonianEuler::Tags::MomentumDensity<Dim>;

  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

  static constexpr double min_density_allowed = 1.0e-18;
  static constexpr double min_pressure_allowed = 1.0e-18;

 public:
  using return_tags = tmpl::list<::Tags::Variables<
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<MassDensityCons, MomentumDensity,
                                              EnergyDensity>>,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::DataForRdmpTci,
                 evolution::dg::subcell::Tags::SubcellOptions<Dim>>;

  template <size_t ThermodynamicDim>
  static std::tuple<bool, evolution::dg::subcell::RdmpTciData> apply(
      gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
          subcell_grid_prim_vars,
      const Variables<tmpl::list<MassDensityCons, MomentumDensity,
                                 EnergyDensity>>& subcell_vars,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
      const evolution::dg::subcell::RdmpTciData& past_rdmp_tci_data,
      const evolution::dg::subcell::SubcellOptions& subcell_options,
      double persson_exponentconst, bool need_rdmp_data_only);
};
}  // namespace NewtonianEuler::subcell
