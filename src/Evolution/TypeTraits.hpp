// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Evolution/Protocols.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace evolution {
/// @{
/// Helper metafunction that checks if the class `T` is marked as numeric
/// initial data.
template <typename T>
using is_numeric_initial_data =
    tt::conforms_to<T, protocols::NumericInitialData>;

template <typename T>
constexpr bool is_numeric_initial_data_v =
    tt::conforms_to_v<T, protocols::NumericInitialData>;
/// @}
}  // namespace evolution
