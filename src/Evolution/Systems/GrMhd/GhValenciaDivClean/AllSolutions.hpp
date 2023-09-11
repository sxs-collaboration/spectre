// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Actions/SetInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"

namespace grmhd::GhValenciaDivClean::InitialData {
using analytic_solutions_and_data_list = gh::solutions_including_matter<3>;
using initial_data_list =
    tmpl::flatten<tmpl::list<analytic_solutions_and_data_list,
                   tmpl::list<grmhd::GhValenciaDivClean::NumericInitialData>>>;
} // namespace grmhd::GhValenciaDivClean::InitialData
