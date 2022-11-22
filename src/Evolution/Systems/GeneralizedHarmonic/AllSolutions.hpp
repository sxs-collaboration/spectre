// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "PointwiseFunctions/AnalyticData/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhGrMhd/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GhRelativisticEuler/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace GeneralizedHarmonic {
/*!
 * \brief All solutions, including matter ones from GRMHD, etc. Used by
 * e.g. boundary conditions.
 */
template <size_t Dim>
using solutions_including_matter = tmpl::append<
    GeneralizedHarmonic::Solutions::all_solutions<Dim>,
    tmpl::conditional_t<
        Dim == 3,
        tmpl::append<
            GeneralizedHarmonic::Solutions::RelativisticEuler::all_solutions,
            GeneralizedHarmonic::Solutions::grmhd::all_solutions,
            GeneralizedHarmonic::AnalyticData::grmhd::all_analytic_data>,
        tmpl::list<>>>;
}  // namespace GeneralizedHarmonic
