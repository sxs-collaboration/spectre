// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Punctures/AmrCriteria/RefineAtPunctures.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/MakeArray.hpp"

namespace Punctures::AmrCriteria {

std::array<amr::Flag, 3> RefineAtPunctures::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<3>& domain, const ElementId<3>& element_id) {
  // Casting down to MultiplePunctures because that's the only background class
  // we have
  const auto& punctures =
      dynamic_cast<const Punctures::AnalyticData::MultiplePunctures&>(
          background)
          .punctures();
  // Split (h-refine) the element if it contains a puncture
  const auto& block = domain.blocks()[element_id.block_id()];
  for (const auto& puncture : punctures) {
    // Check if the puncture is in the block
    const auto block_logical_coords = block_logical_coordinates_single_point(
        tnsr::I<double, 3>{puncture.position}, block);
    if (not block_logical_coords.has_value()) {
      continue;
    }
    // Check if the puncture is in the element
    if (element_logical_coordinates(*block_logical_coords, element_id)) {
      return make_array<3>(amr::Flag::Split);
    }
  }
  return make_array<3>(amr::Flag::DoNothing);
}

PUP::able::PUP_ID RefineAtPunctures::my_PUP_ID = 0;  // NOLINT
}  // namespace Punctures::AmrCriteria
