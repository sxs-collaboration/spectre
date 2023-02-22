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
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Base tag to retrieve the grid frame centers of objects in the domain
/// corresponding to the `ObjectLabel`.
template <ObjectLabel Label>
struct ObjectCenter : db::BaseTag {};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The grid frame center of the excision sphere for the given object.
///
/// Even though this can easily be retrieved from the domain, we add it as its
/// own tag so we can access it through the base tag. This way, other things
/// (like the control system) can grab the center and be agnostic to what the
/// object actually is.
template <ObjectLabel Label>
struct ExcisionCenter : ObjectCenter<Label>, db::SimpleTag {
  using type = tnsr::I<double, 3, Frame::Grid>;
  static std::string name() { return "CenterObject" + get_output(Label); }

  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<3>>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options(
      const std::unique_ptr<::DomainCreator<3>>& domain_creator);
};
}  // namespace domain::Tags
