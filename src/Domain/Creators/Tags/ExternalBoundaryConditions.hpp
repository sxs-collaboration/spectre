// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/DirectionMap.hpp"

namespace domain::Tags {

/*!
 * The boundary conditions to be applied at external boundaries. Holds an entry
 * per block, and a boundary condition per external direction.
 */
template <size_t Dim>
struct ExternalBoundaryConditions : db::SimpleTag {
  using type = std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->external_boundary_conditions();
  }
};

}  // namespace domain::Tags
