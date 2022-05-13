// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/ResourceInfo.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
namespace OptionTags {
/// \ingroup ParallelGroup
/// Options group for resource allocation
template <typename Metavariables>
struct ResourceInfo {
  using type = Parallel::ResourceInfo<Metavariables>;
  static constexpr Options::String help = {
      "Options for allocating resources. This information will be used when "
      "placing Array and Singleton parallel components on the requested "
      "resources."};
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup ParallelGroup
/// Tag that tells whether to avoid placing Singleton and Array components
/// on the global proc 0
struct AvoidGlobalProc0 : db::SimpleTag {
  using type = bool;
  template <typename Metavariables>
  using option_tags =
      tmpl::list<Parallel::OptionTags::ResourceInfo<Metavariables>>;

  static constexpr bool pass_metavariables = true;
  template <typename Metavariables>
  static bool create_from_options(
      const Parallel::ResourceInfo<Metavariables>& resource_info) {
    return resource_info.avoid_global_proc_0();
  }
};

/// \ingroup ParallelGroup
/// Tag that holds resource info about a singleton.
template <typename ParallelComponent>
struct SingletonInfo : db::SimpleTag {
  using type = Parallel::SingletonInfoHolder<ParallelComponent>;
  static std::string name() { return pretty_type::name<ParallelComponent>(); }

  static constexpr bool pass_metavariables = true;

  template <typename Metavariables>
  using option_tags =
      tmpl::list<Parallel::OptionTags::ResourceInfo<Metavariables>>;

  template <typename Metavariables>
  static type create_from_options(
      const Parallel::ResourceInfo<Metavariables>& resource_info) {
    return resource_info.template get_singleton_info<ParallelComponent>();
  }
};
}  // namespace Tags
}  // namespace Parallel
