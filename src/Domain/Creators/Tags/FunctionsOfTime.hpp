// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
struct FunctionOfTime;
}
/// \endcond

namespace domain::Tags {
/// \brief The FunctionsOfTime initialized from a DomainCreator
struct FunctionsOfTimeInitialize : FunctionsOfTime, db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;

  static constexpr bool pass_metavariables = true;

  static std::string name() { return "FunctionsOfTime"; }

  template <typename Metavariables>
  using option_tags =
      tmpl::list<domain::OptionTags::DomainCreator<Metavariables::volume_dim>>;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Metavariables::volume_dim>>&
          domain_creator) {
    return domain_creator->functions_of_time();
  }
};
}  // namespace domain::Tags
