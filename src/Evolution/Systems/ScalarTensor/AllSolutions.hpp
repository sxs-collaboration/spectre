// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ScalarTensor/Actions/SetInitialData.hpp"
#include "PointwiseFunctions/AnalyticData/GhScalarTensor/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::InitialData {
// These are solutions that can be used for analytic prescriptions
using analytic_solutions_and_data_list =
    gh::ScalarTensor::AnalyticData::all_analytic_data;
using initial_data_list = tmpl::push_back<analytic_solutions_and_data_list,
                                          ScalarTensor::NumericInitialData>;
}  // namespace ScalarTensor::InitialData
