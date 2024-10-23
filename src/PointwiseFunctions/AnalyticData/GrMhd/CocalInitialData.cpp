// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/CocalInitialData.hpp"

#include <memory>
#include <pup.h>

namespace grmhd::AnalyticData {

CocalInitialData::CocalInitialData(std::string data_directory,
                                   const double electron_fraction,
                                   const CocalIdType id_type)
    : data_directory_(std::move(data_directory)),
      electron_fraction_(electron_fraction),
      id_type_(id_type) {}

CocalInitialData::CocalInitialData(const CocalInitialData& rhs)
    : evolution::initial_data::InitialData(rhs) {
  *this = rhs;
}

CocalInitialData& CocalInitialData::operator=(const CocalInitialData& rhs) {
  data_directory_ = rhs.data_directory_;
  electron_fraction_ = rhs.electron_fraction_;
  id_type_ = rhs.id_type_;
  return *this;
}

CocalInitialData::CocalInitialData(CocalInitialData&& rhs)
    : evolution::initial_data::InitialData(rhs) {
  *this = rhs;
}

CocalInitialData& CocalInitialData::operator=(CocalInitialData&& rhs) {
  data_directory_ = std::move(rhs.data_directory_);
  electron_fraction_ = rhs.electron_fraction_;
  id_type_ = rhs.id_type_;
  return *this;
}

std::unique_ptr<evolution::initial_data::InitialData>
CocalInitialData::get_clone() const {
  return std::make_unique<CocalInitialData>(*this);
}

CocalInitialData::CocalInitialData(CkMigrateMessage* msg) : InitialData(msg) {}

void CocalInitialData::pup(PUP::er& p) {
  InitialData::pup(p);
  p | data_directory_;
  p | electron_fraction_;
  p | id_type_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID CocalInitialData::my_PUP_ID = 0;

}  // namespace grmhd::AnalyticData
