// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
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
 * - if the minimum density or pressure on either the DG or subcell mesh are
 *   below \f$10^{-18}\f$, marks the element as troubled and returns. We check
 *   both the FD and DG grids since when a discontinuity is inside the element
 *   oscillations in the DG solution can result in negative values that aren't
 *   present in the FD solution.
 * - runs the Persson TCI on the density and pressure on the DG grid. The reason
 *   for applying the Persson TCI to both the density and pressure is to flag
 *   cells at contact discontinuities. The Persson TCI only works with
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

  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

  static constexpr double min_density_allowed = 1.0e-18;
  static constexpr double min_pressure_allowed = 1.0e-18;

 public:
  using return_tags = tmpl::list<::Tags::Variables<
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>>;
  using argument_tags =
      tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                 Inactive<MassDensityCons>, Inactive<MomentumDensity>,
                 Inactive<EnergyDensity>, hydro::Tags::EquationOfStateBase,
                 domain::Tags::Mesh<Dim>>;

  template <size_t ThermodynamicDim>
  static bool apply(
      gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
          subcell_grid_prim_vars,
      const Scalar<DataVector>& subcell_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& subcell_momentum_density,
      const Scalar<DataVector>& subcell_energy_density,
      const Scalar<DataVector>& dg_mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& dg_momentum_density,
      const Scalar<DataVector>& dg_energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Mesh<Dim>& dg_mesh, double persson_exponent) noexcept;
};
}  // namespace NewtonianEuler::subcell
