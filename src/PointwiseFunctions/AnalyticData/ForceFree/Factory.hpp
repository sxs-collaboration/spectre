// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ForceFree/SimulationType.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/FfeBreakdown.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree::AnalyticData {
/*!
 * \brief Typelist of all analytic data of GRFFE evolution system
 */
using all_data = tmpl::list<FfeBreakdown>;

template <SimulationType simulation_type>
using available_data = tmpl::append<
    tmpl::conditional_t<simulation_type == SimulationType::FreeSpace,
                        tmpl::list<FfeBreakdown>, tmpl::list<>>,
    tmpl::conditional_t<simulation_type ==
                            SimulationType::IsolatedNsMagnetosphere,
                        tmpl::list<>, tmpl::list<>>>;
}  // namespace ForceFree::AnalyticData
