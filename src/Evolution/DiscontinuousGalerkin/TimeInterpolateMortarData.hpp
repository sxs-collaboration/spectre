// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/MaybeOwned.hpp"

/// \cond
class TimeStepId;
namespace evolution::dg {
template <size_t Dim>
struct MortarData;
}  // namespace evolution::dg
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace TimeSteppers {
template <typename Data>
class ConstBoundaryHistorySide;
}  // namespace TimeSteppers
/// \endcond

namespace evolution::dg {
/// Interpolate MortarData in time.
///
/// Each vector in the history is interpolated using $(k-1)$th order
/// polynomial interpolation, where $k$ is the integration order of
/// the most recent step in the history.  The interpolation stencil is
/// centered as well as possible on the goal time given the available
/// entries.  Only full step data form the history is used; substep
/// entries are ignored.  Mesh data must be the same for all entries
/// in the stencil.
///
/// If \p include_geometric_data is false, `MortarData` fields other
/// than `mortar_mesh` and `mortar_data` will be ignored during
/// interpolation.  These fields may still be set in the returned
/// value if no interpolation occurs.
///
/// The return-by-value overload will avoid copying data if the
/// requested time is present in the history, but will allocate if it
/// is not, while the return by reference overload will always copy
/// but will not allocate if the return object is already sized
/// correctly.
/// @{
template <size_t Dim>
void time_interpolate_mortar_data(
    gsl::not_null<MortarData<Dim>*> interpolated,
    const TimeSteppers::ConstBoundaryHistorySide<MortarData<Dim>>& history,
    double time, bool include_geometric_data);

template <size_t Dim>
MaybeOwned<const MortarData<Dim>> time_interpolate_mortar_data(
    const TimeSteppers::ConstBoundaryHistorySide<MortarData<Dim>>& history,
    double time, bool include_geometric_data);
/// @}
}  // namespace evolution::dg
