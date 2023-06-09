// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity {

namespace OptionTags {
template <size_t Dim>
struct ConstitutiveRelation : db::SimpleTag {
  static std::string name() { return "Material"; }
  static constexpr Options::String help =
      "The constitutive relation of the elastic material.";
  using type =
      std::unique_ptr<ConstitutiveRelations::ConstitutiveRelation<Dim>>;
};
}  // namespace OptionTags

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

  using option_tags = tmpl::list<OptionTags::ConstitutiveRelation<Dim>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) {
    return deserialize<type>(serialize<type>(value).data());
  }
};

}  // namespace Tags
}  // namespace Elasticity
