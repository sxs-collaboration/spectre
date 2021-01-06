// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // For Tags::Normalized
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace Tags {
/// Compute the characteristic speeds on the moving mesh given the
/// characteristic speeds if the mesh were stationary.
///
/// \note Assumes that `typename CharSpeedsComputeTag::return_type` is a
/// `std::array<DataVector, NumberOfCharSpeeds>`
template <typename CharSpeedsComputeTag, size_t Dim>
struct CharSpeedCompute : CharSpeedsComputeTag::base, db::ComputeTag {
  using base = typename CharSpeedsComputeTag::base;
  using return_type = typename CharSpeedsComputeTag::return_type;

  template <typename... Ts, typename T, size_t NumberOfCharSpeeds>
  static void function(
      const gsl::not_null<std::array<T, NumberOfCharSpeeds>*> result,
      const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
          grid_velocity,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
      const Ts&... ts) noexcept {
    // Note that while the CharSpeedsComputeTag almost certainly also needs the
    // unit normal covector for computing the original characteristic speeds, we
    // don't know which of the `ts` it is, and thus we need the unit normal
    // covector to be passed explicitly.
    CharSpeedsComputeTag::function(result, ts...);
    if (grid_velocity.has_value()) {
      const Scalar<DataVector> normal_dot_velocity =
          dot_product(*grid_velocity, unit_normal_covector);
      for (size_t i = 0; i < result->size(); ++i) {
        gsl::at(*result, i) -= get(normal_dot_velocity);
      }
    }
  }

  using argument_tags =
      tmpl::push_front<typename CharSpeedsComputeTag::argument_tags,
                       MeshVelocity<Dim, Frame::Inertial>,
                       ::Tags::Normalized<UnnormalizedFaceNormal<Dim>>>;
  using volume_tags = get_volume_tags<CharSpeedsComputeTag>;
};
}  // namespace Tags
}  // namespace domain
