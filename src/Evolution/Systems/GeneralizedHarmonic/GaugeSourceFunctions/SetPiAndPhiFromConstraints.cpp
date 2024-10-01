// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.hpp"

#include <cstddef>

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/SetPiAndPhiFromConstraints.tpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Factory.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace gh::gauges {
void SetPiAndPhiFromConstraintsCacheMutator::apply(
    const gsl::not_null<bool*> value, const bool new_value) {
  *value = new_value;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                 \
  template class SetPiAndPhiFromConstraints< \
      gh::Solutions::all_solutions<DIM(data)>, DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace gh::gauges
