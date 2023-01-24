// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/ForceFree/FfeBreakdown.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::AnalyticData {
/// \brief List of all analytic data
using all_data = tmpl::list<FfeBreakdown>;
}  // namespace ForceFree::AnalyticData
