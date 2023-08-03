// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::grmhd::Solutions {
/// GRMHD solutions wrapped for GH

/// \brief List of all analytic solutions
using all_solutions =
    tmpl::list<gh::Solutions::WrappedGr<::grmhd::Solutions::BondiMichel>>;

}  // namespace gh::grmhd::Solutions
