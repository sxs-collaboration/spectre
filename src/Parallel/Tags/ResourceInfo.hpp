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
///
/// We include this tag in addition to Parallel::Tags::ResourceInfo even though
/// this tag's info is contained in Parallel::Tags::ResourceInfo because this
/// tag will inform what the input file looks like
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

/// \ingroup ParallelGroup
/// Tag to retrieve the ResourceInfo.
///
/// \details This tag is meant to be used in the GlobalCache. It can be created
/// from options, but since it must be specially created inside Main and
/// assigned to the GlobalCache after all the global cache tags are created from
/// options, the return of `create_from_options` is just a default constructed
/// Parallel::ResourceInfo.
template <typename Metavariables>
struct ResourceInfo : db::SimpleTag {
  using type = Parallel::ResourceInfo<Metavariables>;

  static constexpr bool pass_metavariables = true;

  // The option tag is needed to force it's inclusion in the input file, but we
  // don't actually use its value to create from options. See docs for this tag
  // for an explanation.
  template <typename Metavars>
  using option_tags = tmpl::list<Parallel::OptionTags::ResourceInfo<Metavars>>;

  template <typename Metavars>
  static type create_from_options(
      const Parallel::ResourceInfo<Metavars>& /*resource_info*/) {
    return type{};
  }
};
}  // namespace Tags
}  // namespace Parallel
