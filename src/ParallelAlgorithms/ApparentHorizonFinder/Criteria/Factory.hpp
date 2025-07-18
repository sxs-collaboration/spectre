// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Criteria/Residual.hpp"
#include "Utilities/TMPL.hpp"

namespace ah::Criteria {
using standard_criteria = tmpl::list<Residual>;
}  // namespace ah::Criteria
