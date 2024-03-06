// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Loehner.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Persson.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/TruncationError.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {

/*!
 * \brief AMR criteria that are generally applicable
 *
 * \tparam Dim the spatial dimension
 * \tparam TensorTags the tensor fields to monitor
 */
template <size_t Dim, typename TensorTags>
using standard_criteria = tmpl::list<
    // p-AMR criteria
    ::amr::Criteria::IncreaseResolution<Dim>,
    ::amr::Criteria::TruncationError<Dim, TensorTags>,
    // h-AMR criteria
    ::amr::Criteria::Loehner<Dim, TensorTags>,
    ::amr::Criteria::Persson<Dim, TensorTags>,
    // Criteria for testing or experimenting
    ::amr::Criteria::DriveToTarget<Dim>, ::amr::Criteria::Random>;

}  // namespace amr::Criteria
