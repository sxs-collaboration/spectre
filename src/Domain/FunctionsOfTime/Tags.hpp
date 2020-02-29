// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/OptionTags.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace Tags {
/// The functions of time obtained from a domain creator
template <size_t Dim>
struct InitialFunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
  using option_tags = tmpl::list<domain::OptionTags::DomainCreator<Dim>>;

  template <typename Metavariables>
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept {
    return domain_creator->functions_of_time();
  }
};

/// The functions of time
struct FunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
};
}  // namespace Tags
}  // namespace domain
