// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

/// \cond
namespace Elasticity {
namespace Tags {
template <size_t Dim>
struct Stress;
}  // namespace Tags
}  // namespace Elasticity
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
/// \endcond

namespace Elasticity {
namespace Tags {

/// Base tag for the constitutive relation
struct ConstitutiveRelationBase : db::BaseTag {};

template <typename ConstitutiveRelationType>
struct ConstitutiveRelation : ConstitutiveRelationBase, db::SimpleTag {
  static constexpr OptionString help = {
      "The constitutive relation of the elastic material"};
  using type = ConstitutiveRelationType;
  static std::string name() noexcept { return "Material"; }
};

}  // namespace Tags
}  // namespace Elasticity
