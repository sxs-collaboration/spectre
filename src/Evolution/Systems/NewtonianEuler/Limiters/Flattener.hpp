// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Mesh;

namespace EquationsOfState {
template <bool, size_t>
class EquationOfState;
}  // namespace EquationsOfState
/// \endcond

namespace NewtonianEuler::Limiters {

/// \ingroup LimitersGroup
/// \brief Encodes the action taken by `flatten_solution`
enum class FlattenerAction {
  NoOp = 0,
  ScaledSolution = 1,
  SetSolutionToMean = 2,
};

/// \ingroup LimitersGroup
/// \brief Scale a NewtonianEuler solution around its mean to remove pointwise
/// positivity violations.
///
/// If the solution has points with negative density, scales the solution
/// to make these points positive again. For each component \f$u\f$ of the
/// solution, the scaling takes the form
/// \f$u \to \bar{u} + \theta (u - \bar{u})\f$,
/// where \f$\bar{u}\f$ is the cell-average value of \f$u\f$, and \f$\theta\f$
/// is a factor less than 1, chosen to restore positive density.
/// The cell averages in this implementation are computed in inertial
/// coordinates, so the flattener is conservative even on deformed grids.
///
/// A scaling of this form used to restore positivity is usually called a
/// flattener (we use this name) or a bounds-preserving filter. Note that the
/// scaling approach only works if the cell-averaged solution is itself
/// physical, in other words, if the cell-averaged density is positive.
///
/// After checking for (and correcting) negative densities, if the equation of
/// state is two-dimensional, then the pressure is also checked for positivity.
/// If negative pressures are found, each solution component is set to its mean
/// (this is equivalent to \f$\theta = 0\f$ in the scaling form above).
/// In principle, a less aggressive scaling could be used, but solving for the
/// correct \f$\theta\f$ in this case is more involved.
template <size_t VolumeDim, size_t ThermodynamicDim>
FlattenerAction flatten_solution(
    gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept;

}  // namespace NewtonianEuler::Limiters
