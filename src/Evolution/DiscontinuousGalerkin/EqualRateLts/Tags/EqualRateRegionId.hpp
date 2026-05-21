// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/DiscontinuousGalerkin/EqualRateLts/EqualRateRegions.hpp"

namespace evolution::dg::Tags {
/// Tag for labeling `Parallel::Section`s for `EqualRateRegions`.
struct EqualRateRegionId {
  using type = evolution::dg::EqualRateRegionId;
};
}  // namespace evolution::dg::Tags
