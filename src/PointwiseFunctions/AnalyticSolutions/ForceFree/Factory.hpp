// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ForceFree/SimulationType.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::Solutions {
/*!
 * \brief Typelist of all analytic solutions of GRFFE evolution system
 */
using all_solutions = tmpl::list<AlfvenWave, FastWave>;

template <SimulationType simulation_type>
using available_solutions = tmpl::append<
    tmpl::conditional_t<simulation_type == SimulationType::FreeSpace,
                        tmpl::list<AlfvenWave, FastWave>, tmpl::list<>>,
    tmpl::conditional_t<simulation_type ==
                            SimulationType::IsolatedNsMagnetosphere,
                        tmpl::list<>, tmpl::list<>>>;
}  // namespace ForceFree::Solutions
