// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"

template <size_t VolumeDim>
class Domain;
template <size_t VolumeDim>
class ElementId;

namespace ah {
/*!
 * \brief Interpolate volume data from any new elements to the target points
 */
template <typename Fr>
void interpolate_volume_data(
    gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration_storage,
    gsl::not_null<
        std::unordered_map<ElementId<3>, ah::Storage::VolumeVariables<Fr>>*>
        all_volume_variables,
    const LinkedMessageId<double>& time, const Domain<3>& domain,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace ah
