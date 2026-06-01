// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/SelfForce/Scalar/AmrCriteria/RefineAtPuncture.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/AnalyticData/CircularOrbit.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeArray.hpp"

namespace ScalarSelfForce::AmrCriteria {

std::array<amr::Flag, 2> RefineAtPuncture::impl(
    const elliptic::analytic_data::Background& background,
    const Domain<2>& domain, const ElementId<2>& element_id) {
  const auto circular_orbit_ptr =
      dynamic_cast<const ScalarSelfForce::AnalyticData::CircularOrbit*>(
          &background);
  if (circular_orbit_ptr == nullptr) {
    ERROR(
        "RefineAtPuncture only works with 'CircularOrbit'. "
        "See ScalarSelfForce::AmrCriteria::RefineAtPuncture for details.");
  }
  const auto puncture_position = circular_orbit_ptr->puncture_position();
  // Split (h-refine) the element if it contains the puncture
  const auto& block = domain.blocks()[element_id.block_id()];
  // Check if the puncture is in the block
  auto block_logical_coords =
      block_logical_coordinates_single_point(puncture_position, block);
  if (not block_logical_coords.has_value()) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  for (size_t d=0; d<2; ++d) {
    if (abs(block_logical_coords->get(d)) < 1e-10) {
      block_logical_coords->get(d) = 0.;
    }
  }
  if (not element_logical_coordinates(*block_logical_coords, element_id)) {
    return make_array<2>(amr::Flag::DoNothing);
  }
  return make_array<2>(amr::Flag::Split);
}

PUP::able::PUP_ID RefineAtPuncture::my_PUP_ID = 0;  // NOLINT

}  // namespace ScalarSelfForce::AmrCriteria
