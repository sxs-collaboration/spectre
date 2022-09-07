// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::Solutions {
/// \brief List of all analytic solutions
using all_solutions = tmpl::list<Bump, Linear, Step>;
}  // namespace Burgers::Solutions
