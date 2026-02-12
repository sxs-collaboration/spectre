// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.hpp"

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
template <size_t Dim>
void CleanMortarHistory<Dim>::apply(
    const gsl::not_null<DirectionalIdMap<
        Dim, TimeSteppers::BoundaryHistory<::evolution::dg::MortarData<Dim>,
                                           ::evolution::dg::MortarData<Dim>,
                                           DataVector>>*>
        history,
    const LtsTimeStepper& time_stepper,
    const DirectionalIdMap<Dim, MortarInfo<Dim>>& mortar_info) {
  for (auto& [mortar_id, hist] : *history) {
    const auto time_stepping_policy =
        mortar_info.at(mortar_id).time_stepping_policy();
    switch (time_stepping_policy) {
      case TimeSteppingPolicy::EqualRate:
        break;
      case TimeSteppingPolicy::Conservative:
        time_stepper.clean_boundary_history(make_not_null(&hist));
        break;
      default:
        ERROR("Unhandled TimeSteppingPolicy: " << time_stepping_policy);
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct CleanMortarHistory<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
