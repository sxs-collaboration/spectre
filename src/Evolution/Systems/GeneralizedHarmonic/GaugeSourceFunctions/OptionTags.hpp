// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"

namespace GeneralizedHarmonic::gauges {
/// \brief Input option tags for the generalized harmonic evolution system
namespace OptionTags {
/// Gauge-related option tags for the GeneralizedHarmonic evolution system.
struct GaugeGroup {
  static std::string name() noexcept { return "Gauge"; }
  static constexpr OptionString help{
      "Gauge-specific options for the GH evolution system"};
  using group = GeneralizedHarmonic::OptionTags::Group;
};
}  // namespace OptionTags
}  // namespace GeneralizedHarmonic::gauges
