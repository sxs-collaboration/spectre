// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace gh::gauges {
#define GH_GAUGE_DISPATCH_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GH_GAUGE_DISPATCH_SOLUTION(data) \
  gh::Solutions::all_solutions<BOOST_PP_TUPLE_ELEM(0, data)>

GENERATE_INSTANTIATIONS(INSTANTIATE_GH_GAUGE_DISPATCH, (1, 2, 3))

#undef GH_GAUGE_DISPATCH_SOLUTION
#undef GH_GAUGE_DISPATCH_DIM
}  // namespace gh::gauges
