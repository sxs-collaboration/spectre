// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/HorizonManagerComponentActions.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"

/// \ingroup SurfacesGroup
namespace ah {

// Holds all the options for a DataInterpolator, so that
// they can be specified under a DataInterpolator heading in the input file.
namespace DataInterpolator_detail {
struct OptionHolder {
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr OptionString help = {"Verbosity"};
    static type default_value() { return ::Verbosity::Quiet; }
  };
  using options = tmpl::list<Verbosity>;
  static constexpr OptionString help = {"Options for horizon manager"};

  explicit OptionHolder(::Verbosity verbosity_in) : verbosity(verbosity_in) {}

  OptionHolder() = default;
  OptionHolder(const OptionHolder& /*rhs*/) = default;
  OptionHolder& operator=(const OptionHolder& /*rhs*/) = default;
  OptionHolder(OptionHolder&& /*rhs*/) noexcept = default;
  OptionHolder& operator=(OptionHolder&& /*rhs*/) noexcept = default;
  ~OptionHolder() = default;

  ::Verbosity verbosity{::Verbosity::Quiet};
};
}  // namespace DataInterpolator_detail

namespace OptionTags {
struct DataInterpolator {
  using type = DataInterpolator_detail::OptionHolder;
  static constexpr OptionString help = {"Options for DataInterpolator"};
};
}  // namespace OptionTags

/// Group component responsible for interpolating data to ah::Finders.
/// Metavariables must contain (in addition to the usual things it contains)
/// a type alias called horizon_tags that is a tmpl::list of AhTags,
/// where each AhTag is a struct that contains:
///         - a static function 'label()' returning std::string or const char*.
///         - a type alias 'frame' to the ::Frame of the horizon.
///         - a type alias 'option_tag' to something in Horizon::OptionTags.
///         - a type alias 'convergence_hook' to a struct with a static function
///       void apply(const Strahlkorper<AhTag::frame>&, const Time &,
///                  const Parallel::ConstGlobalCache<Metavariables>&) noexcept;
///           that the ah::Finder will call when it converges.
/// There should be one AhTag for each ah::Finder.
template <class Metavariables>
struct DataInterpolator {
  using chare_type = Parallel::Algorithms::Group;
  using const_global_cache_tag_list = tmpl::list<>;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename Actions::DataInterpolator::Initialize::
                                   return_tag_list<Metavariables>>;
  using options = tmpl::list<OptionTags::DataInterpolator>;
  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      DataInterpolator_detail::OptionHolder&& option_holder);
  static void execute_next_phase(typename Metavariables::Phase /*next_phase*/,
                                 const Parallel::CProxy_ConstGlobalCache<
                                     Metavariables>& /*global_cache*/){};
};

template <class Metavariables>
void DataInterpolator<Metavariables>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
    DataInterpolator_detail::OptionHolder&& option_holder) {
  auto& my_proxy = Parallel::get_parallel_component<DataInterpolator>(
      *(global_cache.ckLocalBranch()));
  Parallel::simple_action<Actions::DataInterpolator::Initialize>(
      my_proxy, option_holder.verbosity);
}
}  // namespace ah
