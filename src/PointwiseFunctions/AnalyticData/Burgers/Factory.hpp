// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::AnalyticData {
/// \brief List of all analytic data
using all_data = tmpl::list<Sinusoid>;
}  // namespace Burgers::AnalyticData
