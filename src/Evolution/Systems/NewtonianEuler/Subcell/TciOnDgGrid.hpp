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
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename T>
class Variables;
/// \endcond

namespace NewtonianEuler::subcell {
/*!
 * \brief Troubled-cell indicator applied to the DG solution.
 *
 * Computes the primitive variables on the DG grid, mutating them in the
 * DataBox. Then,
 * - if the minimum density or pressure are below \f$10^{-18}\f$ (the arbitrary
 *   threshold used to signal "negative" density and pressure), marks the
 *   element as troubled and returns
 * - runs the Persson TCI on the density and pressure. The reason for applying
 *   the Persson TCI to both the density and pressure is to flag cells at
 *   contact discontinuities.
 */
template <size_t Dim>
class TciOnDgGrid {
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
      tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity,
                 hydro::Tags::EquationOfStateBase, domain::Tags::Mesh<Dim>>;

  template <size_t ThermodynamicDim>
  static bool apply(
      const gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
          dg_prim_vars,
      const Scalar<DataVector>& mass_density,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
      const Mesh<Dim>& dg_mesh, const double persson_exponent);
};
}  // namespace NewtonianEuler::subcell
