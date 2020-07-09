// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
/// \endcond

namespace Burgers {
namespace Tags {
/// Computes the characteristic speeds
struct CharacteristicSpeedsCompute : CharacteristicSpeeds, db::ComputeTag {
  static std::string name() noexcept { return "CharacteristicSpeeds"; }

  using argument_tags =
      tmpl::list<Tags::U, domain::Tags::UnnormalizedFaceNormal<1>>;

  using return_type = std::array<DataVector, 1>;
  static void function(gsl::not_null<return_type*> result,
                       const Scalar<DataVector>& u,
                       const tnsr::i<DataVector, 1>& normal) noexcept;
};
}  // namespace Tags

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::U>;
  static double apply(const Scalar<DataVector>& u) noexcept;
};
}  // namespace Burgers
