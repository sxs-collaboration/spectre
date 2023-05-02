// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace gh {
/*!
 * \brief All solutions, including matter ones from GRMHD, etc. Used by
 * e.g. boundary conditions.
 */
template <size_t Dim>
using solutions_including_matter = tmpl::append<
    gh::Solutions::all_solutions<Dim>,
    tmpl::conditional_t<
        Dim == 3,
        tmpl::append<gh::Solutions::RelativisticEuler::all_solutions,
                     gh::Solutions::grmhd::all_solutions,
                     gh::AnalyticData::grmhd::all_analytic_data>,
        tmpl::list<>>>;
}  // namespace gh
