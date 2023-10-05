// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/PolarMagnetizedFmDisk.hpp"

#include <pup.h>
#include <utility>


namespace grmhd::AnalyticData {

PolarMagnetizedFmDisk::PolarMagnetizedFmDisk(
    MagnetizedFmDisk fm_disk, grmhd::AnalyticData::SphericalTorus torus_map)
    : fm_disk_(std::move(fm_disk)), torus_map_(std::move(torus_map)) {}

std::unique_ptr<evolution::initial_data::InitialData>
PolarMagnetizedFmDisk::get_clone() const {
  return std::make_unique<PolarMagnetizedFmDisk>(*this);
}

PolarMagnetizedFmDisk::PolarMagnetizedFmDisk(CkMigrateMessage* msg)
    : fm_disk_(msg) {}

void PolarMagnetizedFmDisk::pup(PUP::er& p) {
  p | fm_disk_;
  p | torus_map_;
}

PUP::able::PUP_ID PolarMagnetizedFmDisk::my_PUP_ID = 0; // NOLINT

bool operator==(const PolarMagnetizedFmDisk& lhs,
                const PolarMagnetizedFmDisk& rhs) {
  return lhs.fm_disk_ == rhs.fm_disk_ and lhs.torus_map_ == rhs.torus_map_;
}

bool operator!=(const PolarMagnetizedFmDisk& lhs,
                const PolarMagnetizedFmDisk& rhs) {
  return not(lhs == rhs);
}
}  // namespace grmhd::AnalyticData
