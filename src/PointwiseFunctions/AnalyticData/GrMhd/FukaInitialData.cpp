// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/FukaInitialData.hpp"

#include <memory>
#include <pup.h>

namespace grmhd::AnalyticData {

FukaInitialData::FukaInitialData(std::string info_filename,
                                 const double electron_fraction)
    : info_filename_(std::move(info_filename)),
      electron_fraction_(electron_fraction) {}

FukaInitialData::FukaInitialData(const FukaInitialData& rhs)
    : evolution::initial_data::InitialData(rhs) {
  *this = rhs;
}

FukaInitialData& FukaInitialData::operator=(const FukaInitialData& rhs) {
  info_filename_ = rhs.info_filename_;
  electron_fraction_ = rhs.electron_fraction_;
  return *this;
}

FukaInitialData::FukaInitialData(FukaInitialData&& rhs)
    : evolution::initial_data::InitialData(rhs) {
  *this = rhs;
}

FukaInitialData& FukaInitialData::operator=(FukaInitialData&& rhs) {
  info_filename_ = std::move(rhs.info_filename_);
  electron_fraction_ = rhs.electron_fraction_;
  return *this;
}

std::unique_ptr<evolution::initial_data::InitialData>
FukaInitialData::get_clone() const {
  return std::make_unique<FukaInitialData>(*this);
}

FukaInitialData::FukaInitialData(CkMigrateMessage* msg) : InitialData(msg) {}

void FukaInitialData::pup(PUP::er& p) {
  InitialData::pup(p);
  p | info_filename_;
  p | electron_fraction_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID FukaInitialData::my_PUP_ID = 0;

}  // namespace grmhd::AnalyticData
