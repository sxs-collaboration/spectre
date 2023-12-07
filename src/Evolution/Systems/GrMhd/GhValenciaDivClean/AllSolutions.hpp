// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GeneralizedHarmonic/AllSolutions.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Actions/SetInitialData.hpp"

// Check if SpEC is linked and therefore we can load SpEC initial data
#ifdef HAS_SPEC_EXPORTER
#include "PointwiseFunctions/AnalyticData/GrMhd/SpecInitialData.hpp"
using SpecInitialDataList = tmpl::list<grmhd::AnalyticData::SpecInitialData<1>,
                                       grmhd::AnalyticData::SpecInitialData<2>>;
#else
using SpecInitialDataList = NoSuchType;
#endif

// Check if FUKA is linked and therefore we can load FUKA initial data
#ifdef HAS_FUKA_EXPORTER
#include "PointwiseFunctions/AnalyticData/GrMhd/FukaInitialData.hpp"
using FukaInitialDataList = tmpl::list<grmhd::AnalyticData::FukaInitialData>;
#else
using FukaInitialDataList = NoSuchType;
#endif

namespace ghmhd::GhValenciaDivClean::InitialData {
// These are solutions that can be used for analytic prescriptions
using analytic_solutions_and_data_list = gh::ghmhd_solutions;
using initial_data_list = tmpl::flatten<tmpl::list<
    analytic_solutions_and_data_list,
    tmpl::flatten<tmpl::list<
        grmhd::GhValenciaDivClean::NumericInitialData,
        tmpl::conditional_t<std::is_same_v<SpecInitialDataList, NoSuchType>,
                            tmpl::list<>, SpecInitialDataList>,
        tmpl::conditional_t<std::is_same_v<FukaInitialDataList, NoSuchType>,
                            tmpl::list<>, FukaInitialDataList>>>>>;
}  // namespace ghmhd::GhValenciaDivClean::InitialData
