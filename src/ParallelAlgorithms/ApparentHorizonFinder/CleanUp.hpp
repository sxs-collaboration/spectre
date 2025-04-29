// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <set>
#include <unordered_map>

#include "DataStructures/LinkedMessageId.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/Gsl.hpp"

namespace ah {
/*!
 * \brief Cleans up the horizon finder after a horizon find has finished
 *
 * \details Removed the current time from the storage map, adds the current time
 * to the completed times, and then resets the current time. If the completed
 * times have more than 1000 entries, this will limit the size to 1000.
 */
template <typename Fr>
void clean_up_horizon_finder(
    gsl::not_null<std::optional<LinkedMessageId<double>>*>
        current_time_optional,
    gsl::not_null<std::unordered_map<LinkedMessageId<double>,
                                     ah::Storage::SingleTimeStorage<Fr>>*>
        all_storage,
    gsl::not_null<std::set<LinkedMessageId<double>>*> completed_times,
    gsl::not_null<FastFlow*> fast_flow);
}  // namespace ah
