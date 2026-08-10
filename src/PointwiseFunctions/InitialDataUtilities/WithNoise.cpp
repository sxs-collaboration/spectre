// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/InitialDataUtilities/WithNoise.hpp"

#include <pup.h>
#include <pup_stl.h>
#include <random>
#include <utility>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace evolution::initial_data {

WithNoise::WithNoise(const WithNoise& rhs)
    : InitialData(rhs),
      solution_(rhs.solution_ != nullptr ? rhs.solution_->get_clone()
                                         : nullptr),
      amplitude_(rhs.amplitude_),
      seed_(rhs.seed_),
      variables_(rhs.variables_) {}

WithNoise& WithNoise::operator=(const WithNoise& rhs) {
  if (this == &rhs) {
    return *this;
  }
  InitialData::operator=(rhs);
  solution_ = rhs.solution_ != nullptr ? rhs.solution_->get_clone() : nullptr;
  amplitude_ = rhs.amplitude_;
  seed_ = rhs.seed_;
  variables_ = rhs.variables_;
  return *this;
}

WithNoise::WithNoise(
    std::unique_ptr<evolution::initial_data::InitialData> solution,
    const double amplitude, std::optional<size_t> seed,
    std::vector<std::string> variables)
    : solution_(std::move(solution)),
      amplitude_(amplitude),
      seed_(seed.value_or(std::random_device{}())),
      variables_(std::move(variables)) {
  if (dynamic_cast<const WithNoise*>(solution_.get()) != nullptr) {
    ERROR("WithNoise cannot wrap another WithNoise. Nesting is not supported.");
  }
}

WithNoise::WithNoise(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> WithNoise::get_clone()
    const {
  return std::make_unique<WithNoise>(*this);
}

void WithNoise::pup(PUP::er& p) {
  evolution::initial_data::InitialData::pup(p);
  p | solution_;
  p | amplitude_;
  p | seed_;
  p | variables_;
}

bool operator==(const WithNoise& lhs, const WithNoise& rhs) {
  if (lhs.amplitude_ != rhs.amplitude_ or lhs.seed_ != rhs.seed_ or
      lhs.variables_ != rhs.variables_) {
    return false;
  }
  if ((lhs.solution_ == nullptr) != (rhs.solution_ == nullptr)) {
    return false;
  }
  if (lhs.solution_ == nullptr) {
    return true;
  }
  // Compare inner solutions via PUP serialization. Requires Charm++ factory
  // classes to be registered (register_factory_classes_with_charm) before use.
  return serialize(lhs.solution_) == serialize(rhs.solution_);
}

bool operator!=(const WithNoise& lhs, const WithNoise& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID WithNoise::my_PUP_ID = 0;  // NOLINT

}  // namespace evolution::initial_data
