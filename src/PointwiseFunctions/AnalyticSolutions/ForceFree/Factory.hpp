// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/ForceFree/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::Solutions {
/*!
 * \brief Typelist of all analytic solutions of GRFFE evolution system
 */
using all_solutions = tmpl::list<AlfvenWave, FastWave>;
}  // namespace ForceFree::Solutions
