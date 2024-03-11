// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/ForceFree/FfeBreakdown.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/MagnetosphericWald.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/RotatingDipole.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::AnalyticData {
/*!
 * \brief Typelist of all analytic data of GRFFE evolution system
 */
using all_data = tmpl::list<FfeBreakdown, MagnetosphericWald, RotatingDipole>;
}  // namespace ForceFree::AnalyticData
