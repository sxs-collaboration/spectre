// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Kerr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::Solutions {
using all_analytic_solutions = tmpl::list<Flatness, Kerr, Schwarzschild>;
}  // namespace Xcts::Solutions
