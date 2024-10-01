// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TimeDerivative.tpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/AllSolutions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template struct gh::TimeDerivative<
    ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
    3>;

template class gh::gauges::Tags::GaugeAndDerivativeCompute<
    3,
    ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list>;

template class gh::gauges::SetPiAndPhiFromConstraints<
    ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list,
    3>;

namespace gh::gauges {
#define GH_GAUGE_DISPATCH_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GH_GAUGE_DISPATCH_SOLUTION(data) \
  ghmhd::GhValenciaDivClean::InitialData::analytic_solutions_and_data_list

GENERATE_INSTANTIATIONS(INSTANTIATE_GH_GAUGE_DISPATCH, (3))

#undef GH_GAUGE_DISPATCH_SOLUTION
#undef GH_GAUGE_DISPATCH_DIM
}  // namespace gh::gauges
