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

namespace grmhd::ValenciaDivClean::Limiters {

/// \ingroup LimitersGroup
/// \brief Encodes the action taken by `flatten_solution`
enum class FlattenerAction {
  NoOp = 0,
  ScaledSolution = 1,
  SetSolutionToMean = 2,
};

/// \ingroup LimitersGroup
/// \brief Scale a ValenciaDivClean solution around its mean to remove pointwise
/// positivity violations.
///
/// If the solution has points with negative \f${\tilde D}\f$, scales the
/// solution to make these points positive again. For each _hydro_ component
/// \f$u\f$ of the solution, the scaling takes the form \f$u \to \bar{u} +
/// \theta (u - \bar{u})\f$, where \f$\bar{u}\f$ is the cell-average value of
/// \f$u\f$, and \f$\theta\f$ is a factor less than 1, chosen to restore
/// positive density.
///
/// Then, we check the condition on \f${\tilde \tau}\f$ from Foucart's thesis,
/// and if \f${\tilde \tau}\f$ is too small, we set the _hydro_ components to
/// their cell averages.
///
/// Note the magnetic field and divergence-cleaning field are _not_ flattened,
/// because this could lead to large errors in the divergence. The cell
/// averages in this implementation are computed in inertial coordinates, so
/// the flattener is conservative even on deformed grids.
///
/// TODO:
/// How much more checking is needed? Ideally, we'd identify all "bad" solutions
/// in this function, so that the primitive recovery will be successful. In the
/// Newtonian case, this can be achieved by checking that the density and
/// pressure are both positive, but here the checks are more complicated.
/// In addition to checking TildeD and TildeTau as described above, the TildeS
/// condition from Foucart's thesis could also be checked, though it is more
/// expensive. Could even attempt a full primitive recovery to check success,
/// but is that too much? All this is made up, and can likely be improved!
FlattenerAction flatten_solution(
    gsl::not_null<Scalar<DataVector>*> tilde_d,
    gsl::not_null<Scalar<DataVector>*> tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const tnsr::I<DataVector, 3>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian) noexcept;

}  // namespace grmhd::ValenciaDivClean::Limiters
