// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Burgers {
namespace Tags {
/// Computes the characteristic speeds
struct CharacteristicSpeedsCompute : db::ComputeTag {
  static std::string name() noexcept { return "CharacteristicSpeeds"; }

  using argument_tags = tmpl::list<Tags::U>;

  using return_type = std::array<DataVector, 1>;
  static void function(gsl::not_null<return_type*> result,
                       const Scalar<DataVector>& u) noexcept;
};
}  // namespace Tags
}  // namespace Burgers
