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
  // Collect all puncture positions
  tnsr::I<DataVector, 3> puncture_positions{punctures.size()};
  for (size_t i = 0; i < punctures.size(); ++i) {
    for (size_t d = 0; d < 3; ++d) {
      puncture_positions.get(d)[i] = gsl::at(punctures.at(i).position, d);
    }
  }
  // Use inverse coordinate maps to check if any of the punctures is in this
  // element
  const auto block_logical_coords =
      block_logical_coordinates(domain, puncture_positions);
  const auto element_logical_coords =
      element_logical_coordinates({element_id}, block_logical_coords);
  // Split (h-refine) the element if it contains a puncture, else p-refine
  if (element_logical_coords.find(element_id) != element_logical_coords.end()) {
    return make_array<3>(amr::Flag::Split);
  } else {
    return make_array<3>(amr::Flag::IncreaseResolution);
  }
}

PUP::able::PUP_ID RefineAtPunctures::my_PUP_ID = 0;  // NOLINT
}  // namespace Punctures::AmrCriteria
