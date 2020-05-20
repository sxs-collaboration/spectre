// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"

#include <cstddef>
#include <utility>

#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"

namespace Cce {
AnalyticBoundaryDataManager::AnalyticBoundaryDataManager(
    const size_t l_max, const double extraction_radius,
    std::unique_ptr<Solutions::WorldtubeData> generator) noexcept
    : l_max_{l_max},
      generator_{std::move(generator)},
      extraction_radius_{extraction_radius} {}

void AnalyticBoundaryDataManager::pup(PUP::er& p) noexcept {
  p | l_max_;
  p | extraction_radius_;
  p | generator_;
}
}  // namespace Cce
