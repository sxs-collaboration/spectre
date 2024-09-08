// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>

NumericData::NumericData(std::string file_glob, std::string subgroup,
                         int observation_step, bool extrapolate_into_excisions)
    : file_glob_(std::move(file_glob)),
      subgroup_(std::move(subgroup)),
      observation_step_(observation_step),
      extrapolate_into_excisions_(extrapolate_into_excisions) {}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return lhs.file_glob() == rhs.file_glob() and
         lhs.subgroup() == rhs.subgroup() and
         lhs.observation_step() == rhs.observation_step() and
         lhs.extrapolate_into_excisions() == rhs.extrapolate_into_excisions();
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

void NumericData::pup(PUP::er& p) {
  p | file_glob_;
  p | subgroup_;
  p | observation_step_;
  p | extrapolate_into_excisions_;
}

namespace elliptic::analytic_data {

NumericData::NumericData(CkMigrateMessage* m)
    : elliptic::analytic_data::Background(m),
      elliptic::analytic_data::InitialGuess(m) {}

void NumericData::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  ::NumericData::pup(p);
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return static_cast<const ::NumericData&>(lhs) ==
         static_cast<const ::NumericData&>(rhs);
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT

}  // namespace elliptic::analytic_data

namespace evolution::initial_data {

NumericData::NumericData(CkMigrateMessage* m)
    : evolution::initial_data::InitialData(m) {}

std::unique_ptr<evolution::initial_data::InitialData> NumericData::get_clone()
    const {
  return std::make_unique<NumericData>(*this);
}

void NumericData::pup(PUP::er& p) {
  evolution::initial_data::InitialData::pup(p);
  ::NumericData::pup(p);
}

bool operator==(const NumericData& lhs, const NumericData& rhs) {
  return static_cast<const ::NumericData&>(lhs) ==
         static_cast<const ::NumericData&>(rhs);
}

bool operator!=(const NumericData& lhs, const NumericData& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericData::my_PUP_ID = 0;  // NOLINT

}  // namespace evolution::initial_data
