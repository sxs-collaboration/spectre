// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/CollectStepperErrorTolerances.hpp"

#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "Time/RequestsStepperErrorTolerances.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace collect_stepper_error_tolerances_detail {
void process_request(
    const gsl::not_null<
        std::unordered_map<std::type_index, StepperErrorTolerances>*>
        tolerances,
    const RequestsStepperErrorTolerances& tolerance_request) {
  auto new_requests = tolerance_request.tolerances();
  while (not new_requests.empty()) {
    auto request = new_requests.extract(new_requests.begin());
    if (request.mapped().estimates != StepperErrorTolerances::Estimates::None) {
      const auto inserted = tolerances->insert(std::move(request));
      if (not inserted.inserted and
          inserted.node.mapped() != inserted.position->second) {
        ERROR_NO_TRACE(
            "All time stepping error tolerances for one set of variables must "
            "be the same, but found differing values:\n"
            << inserted.node.mapped() << "\n"
            << inserted.position->second);
      }
    }
  }
}
}  // namespace collect_stepper_error_tolerances_detail
