// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/GrMhd/BondiHoyleAccretion.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/CcsnCollapse.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedFmDisk.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::grmhd::AnalyticData {
/// GRMHD analytic data wrapped for GH

/// \brief List of all analytic data
using all_analytic_data = tmpl::list<
    gh::Solutions::WrappedGr<::grmhd::AnalyticData::BondiHoyleAccretion>,
    gh::Solutions::WrappedGr<::grmhd::AnalyticData::CcsnCollapse>,
    gh::Solutions::WrappedGr<::grmhd::AnalyticData::MagnetizedFmDisk>,
    gh::Solutions::WrappedGr<::grmhd::AnalyticData::MagnetizedTovStar>>;
}  // namespace gh::grmhd::AnalyticData
