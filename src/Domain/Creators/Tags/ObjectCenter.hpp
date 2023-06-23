// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
}  // namespace Frame
/// \endcond

namespace domain::Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup ComputationalDomainGroup
 * \brief The grid frame center of the given object.
 *
 * \note Requires that the domain creator has a grid anchor with the name:
 * "Center + get_output(Label)"
 */
template <ObjectLabel Label>
struct ObjectCenter : db::SimpleTag {
  using type = tnsr::I<double, 3, Frame::Grid>;
  static std::string name() { return "ObjectCenter" + get_output(Label); }

  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<3>>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options(
      const std::unique_ptr<::DomainCreator<3>>& domain_creator);
};
}  // namespace domain::Tags
