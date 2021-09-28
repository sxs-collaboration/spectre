// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"

namespace Elasticity {
namespace Tags {

/*!
 * \brief The elastic material's constitutive relation.
 *
 * \see `Elasticity::ConstitutiveRelations::ConstitutiveRelation`
 */
template <size_t Dim>
struct ConstitutiveRelation : db::SimpleTag {
  using type =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
};

/*!
 * \brief Reference the constitutive relation provided by the `ProviderTag`
 *
 * \see `Elasticity::Tags::ConstitutiveRelation`
 */
template <size_t Dim, typename ProviderTag>
struct ConstitutiveRelationReference : ConstitutiveRelation<Dim>,
                                       db::ReferenceTag {
  using base = ConstitutiveRelation<Dim>;
  using parent_tag = ProviderTag;
  using argument_tags = tmpl::list<ProviderTag>;
  template <typename Provider>
  static const ConstitutiveRelations::ConstitutiveRelation<Dim>& get(
      const Provider& provider) {
    return provider.constitutive_relation();
  }
};

}  // namespace Tags
}  // namespace Elasticity
