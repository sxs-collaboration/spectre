// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/MonteCarlo/HomogeneousSphere.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport::MonteCarlo::Solutions {
/// \brief List of all analytic solutions
using all_solutions = tmpl::list<HomogeneousSphere>;
}  // namespace RadiationTransport::MonteCarlo::Solutions
