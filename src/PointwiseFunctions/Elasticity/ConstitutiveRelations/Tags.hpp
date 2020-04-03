// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {
namespace Tags {

/// Base tag for the constitutive relation
struct ConstitutiveRelationBase : db::BaseTag {};

/*!
 * \brief The elastic material's constitutive relation.
 *
 * When constructing from options, copies the constitutive relation from the
 * `Metavariables::constitutive_relation_provider_option_tag` by calling its
 * constructed object's `constitutive_relation()` member function.
 *
 * The constitutive relation can be retrieved from the DataBox using its base
 * `Elasticity::Tags::ConstitutiveRelation` tag.
 *
 * \see `Elasticity::ConstitutiveRelations::ConstitutiveRelation`
 */
template <typename ConstitutiveRelationType>
struct ConstitutiveRelation : ConstitutiveRelationBase, db::SimpleTag {
  using type = ConstitutiveRelationType;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  using option_tags = tmpl::list<
      typename Metavariables::constitutive_relation_provider_option_tag>;
  template <typename Metavariables, typename ProviderType>
  static type create_from_options(const ProviderType& provider) noexcept {
    return provider.constitutive_relation();
  }
};

}  // namespace Tags
}  // namespace Elasticity
