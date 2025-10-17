// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"

/// \cond
template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ah {
/*!
 * \brief Interpolate volume data from any new elements received by the horizon
 * finder to the target points.
 *
 * \details For each new element, the `vars_to_interpolate_to_target` in
 * \p all_volume_variables are interpolated to the target points and stored
 * in \p current_iteration_storage.
 */
template <typename Fr>
bool interpolate_volume_data(
    gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration_storage,
    const ah::Storage::VolumeVariables<Fr>& volume_vars_storage,
    const ElementId<3>& element_id);
}  // namespace ah
