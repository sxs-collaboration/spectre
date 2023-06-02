// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Phase.hpp"

#include <ostream>
#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Parallel {
std::vector<Phase> known_phases() {
  return {Phase::AdjustDomain,
          Phase::CheckDomain,
          Phase::CheckTimeStepperHistory,
          Phase::Cleanup,
          Phase::EvaluateAmrCriteria,
          Phase::Evolve,
          Phase::Execute,
          Phase::Exit,
          Phase::ImportInitialData,
          Phase::Initialization,
          Phase::InitializeInitialDataDependentQuantities,
          Phase::InitializeTimeStepperHistory,
          Phase::LoadBalancing,
          Phase::PostFailureCleanup,
          Phase::Register,
          Phase::RegisterWithElementDataReader,
          Phase::Solve,
          Phase::Testing,
          Phase::WriteCheckpoint};
}

std::ostream& operator<<(std::ostream& os, const Phase& phase) {
  switch (phase) {
    case Parallel::Phase::AdjustDomain:
      return os << "AdjustDomain";
    case Parallel::Phase::CheckDomain:
      return os << "CheckDomain";
    case Parallel::Phase::CheckTimeStepperHistory:
      return os << "CheckTimeStepperHistory";
    case Parallel::Phase::Cleanup:
      return os << "Cleanup";
    case Parallel::Phase::EvaluateAmrCriteria:
      return os << "EvaluateAmrCriteria";
    case Parallel::Phase::Evolve:
      return os << "Evolve";
    case Parallel::Phase::Execute:
      return os << "Execute";
    case Parallel::Phase::Exit:
      return os << "Exit";
    case Parallel::Phase::ImportInitialData:
      return os << "ImportInitialData";
    case Parallel::Phase::Initialization:
      return os << "Initialization";
    case Parallel::Phase::InitializeInitialDataDependentQuantities:
      return os << "InitializeInitialDataDependentQuantities";
    case Parallel::Phase::InitializeTimeStepperHistory:
      return os << "InitializeTimeStepperHistory";
    case Parallel::Phase::LoadBalancing:
      return os << "LoadBalancing";
    case Parallel::Phase::PostFailureCleanup:
      return os << "PostFailureCleanup";
    case Parallel::Phase::Register:
      return os << "Register";
    case Parallel::Phase::RegisterWithElementDataReader:
      return os << "RegisterWithElementDataReader";
    case Parallel::Phase::Solve:
      return os << "Solve";
    case Parallel::Phase::Testing:
      return os << "Testing";
    case Parallel::Phase::WriteCheckpoint:
      return os << "WriteCheckpoint";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Stream operator does not have case for Phase with integral value "
            << static_cast<std::underlying_type_t<Parallel::Phase>>(phase)
            << "\n");
      // LCOV_EXCL_STOP
  }
}
}  // namespace Parallel

template <>
Parallel::Phase Options::create_from_yaml<Parallel::Phase>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  for (const auto phase : Parallel::known_phases()) {
    if (type_read == get_output(phase)) {
      return phase;
    }
  }
  using ::operator<<;
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read << "\" to Parallel::Phase.\nMust be one of "
                  << Parallel::known_phases() << ".");
}
