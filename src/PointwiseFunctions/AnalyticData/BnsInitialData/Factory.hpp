// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/BnsInitialData/SpectreData.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"

namespace BnsInitialData::AnalyticData {
using all_analytic_data = tmpl::list<SpectreData<1>>;
}  // namespace BnsInitialData::AnalyticData
