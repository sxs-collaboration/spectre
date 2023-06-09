// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <ios>
#include <optional>
#include <pup.h>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/TypeTraits.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

/// \cond
namespace Parallel::Tags {
template <typename Component>
struct SingletonInfo;
struct AvoidGlobalProc0;
template <typename Metavariables>
struct ResourceInfo;
}  // namespace Parallel::Tags
/// \endcond

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Holds resource info for a single singleton component
 *
 * \details Holds what proc the singleton is to be placed on and whether that
 * proc should be exclusive, i.e. no array component elements or other
 * singletons placed on that proc. Instead of specifying a proc, the proc can be
 * chosen automatically by using the `Options::Auto` option.
 *
 * The template parameter `Component` is only used to identify which singleton
 * component this SingletonInfoHolder belongs to.
 */
template <typename Component>
struct SingletonInfoHolder {
  struct Proc {
    using type = Options::Auto<int>;
    static constexpr Options::String help = {
        "Proc to put singleton on. This can be determined automatically if "
        "desired by specifying 'Auto' (without quotes)."};
  };

  struct Exclusive {
    using type = bool;
    static constexpr Options::String help = {
        "Reserve this proc for this singleton. No array component elements or "
        "other singleton components will be placed on this proc."};
  };

  using options = tmpl::list<Proc, Exclusive>;
  static constexpr Options::String help = {
      "Resource options for a single singleton."};

  SingletonInfoHolder(std::optional<int> input_proc, const bool input_exclusive,
                      const Options::Context& context = {})
      : exclusive_(input_exclusive) {
    // If there is no value, we don't need to error so use 0 as a comparator
    // in both cases
    if (input_proc.value_or(0) < 0) {
      PARSE_ERROR(
          context,
          "Proc must be a non-negative integer. Please choose another proc.");
    }

    proc_ = input_proc.has_value()
                ? std::optional<size_t>(static_cast<size_t>(input_proc.value()))
                : std::nullopt;
  }

  SingletonInfoHolder() = default;
  SingletonInfoHolder(const SingletonInfoHolder& /*rhs*/) = default;
  SingletonInfoHolder& operator=(const SingletonInfoHolder& /*rhs*/) = default;
  SingletonInfoHolder(SingletonInfoHolder&& /*rhs*/) = default;
  SingletonInfoHolder& operator=(SingletonInfoHolder&& /*rhs*/) = default;
  ~SingletonInfoHolder() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | proc_;
    p | exclusive_;
  };

  /// Proc that singleton is to be placed on. If the optional is a std::nullopt,
  /// then the proc should be chosen automatically.
  std::optional<size_t> proc() const { return proc_; }

  /// Whether or not the singleton wants to be exclusive on the proc.
  bool is_exclusive() const { return exclusive_; }

 private:
  template <typename ParallelComponent>
  friend bool operator==(const SingletonInfoHolder<ParallelComponent>& lhs,
                         const SingletonInfoHolder<ParallelComponent>& rhs);
  // We use size_t here because we want a non-negative integer, but we use int
  // in the option because we want to protect against negative numbers. And a
  // negative size_t is actually a really large value (it wraps around)
  std::optional<size_t> proc_{std::nullopt};
  bool exclusive_{false};
};

template <typename ParallelComponent>
bool operator==(const SingletonInfoHolder<ParallelComponent>& lhs,
                const SingletonInfoHolder<ParallelComponent>& rhs) {
  return lhs.proc_ == rhs.proc_ and lhs.exclusive_ == rhs.exclusive_;
}

template <typename ParallelComponent>
bool operator!=(const SingletonInfoHolder<ParallelComponent>& lhs,
                const SingletonInfoHolder<ParallelComponent>& rhs) {
  return not(lhs == rhs);
}

template <typename ParallelComponents>
struct SingletonPack;

/*!
 * \ingroup ParallelGroup
 * \brief Holds options for a group of singleton components.
 *
 * \details The info for each singleton in the `ParallelComponents` template
 *  pack is stored in an individual `Parallel::SingletonInfoHolder`.
 *
 * You can pass `Auto` as an option for each singleton in an input file and each
 * singleton will be constructed as a default `Parallel::SingletonInfoHolder`.
 */
template <typename... ParallelComponents>
struct SingletonPack<tmpl::list<ParallelComponents...>> {
 private:
  static_assert((Parallel::is_singleton_v<ParallelComponents> and ...),
                "At least one of the parallel components passed to "
                "SingletonPack is not a Singleton.");
  using component_list = tmpl::list<ParallelComponents...>;

  template <typename Component>
  struct LocalTag {
    using type = SingletonInfoHolder<Component>;
  };
  using local_tags =
      tmpl::transform<component_list, tmpl::bind<LocalTag, tmpl::_1>>;

 public:
  template <typename Component>
  struct SingletonOption {
    using type = Options::Auto<SingletonInfoHolder<Component>>;
    static std::string name() { return pretty_type::name<Component>(); }
    static constexpr Options::String help = {
        "Resource options for a specific singleton."};
  };

  using options =
      tmpl::transform<component_list, tmpl::bind<SingletonOption, tmpl::_1>>;
  static constexpr Options::String help = {
      "Resource options for all singletons."};

  SingletonPack(
      const std::optional<
          SingletonInfoHolder<ParallelComponents>>&... singleton_info_holders,
      const Options::Context& /*context*/ = {})
      : procs_(tuples::tagged_tuple_from_typelist<local_tags>(
            singleton_info_holders.value_or(
                SingletonInfoHolder<ParallelComponents>{})...)) {}

  SingletonPack() = default;
  SingletonPack(const SingletonPack& /*rhs*/) = default;
  SingletonPack& operator=(const SingletonPack& /*rhs*/) = default;
  SingletonPack(SingletonPack&& /*rhs*/) = default;
  SingletonPack& operator=(SingletonPack&& /*rhs*/) = default;
  ~SingletonPack() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) { p | procs_; };

  /// Get a const reference to the SingletonInfoHolder for the `Component`
  /// singleton
  template <typename Component>
  const auto& get() const {
    return tuples::get<LocalTag<Component>>(procs_);
  }

 private:
  template <typename... Components>
  friend bool operator==(const SingletonPack<tmpl::list<Components...>>& lhs,
                         const SingletonPack<tmpl::list<Components...>>& rhs);

  tuples::tagged_tuple_from_typelist<local_tags> procs_{};
};

template <typename... Components>
bool operator==(const SingletonPack<tmpl::list<Components...>>& lhs,
                const SingletonPack<tmpl::list<Components...>>& rhs) {
  return lhs.procs_ == rhs.procs_;
}

template <typename... Components>
bool operator!=(const SingletonPack<tmpl::list<Components...>>& lhs,
                const SingletonPack<tmpl::list<Components...>>& rhs) {
  return not(lhs == rhs);
}

namespace detail {
template <typename Metavariables>
using singleton_components =
    tmpl::filter<typename Metavariables::component_list,
                 Parallel::is_singleton<tmpl::_1>>;
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Holds resource info for all singletons and for avoiding placing array
 * elements/singletons on the global proc 0.
 *
 * \details This can be used for placing all singletons in an executable.
 *
 * If you have no singletons, you'll need the following block in the input file
 * (where you can set the value of AvoidGlobalProc0 to true or false):
 *
 * \code {.yaml}
 * ResourceInfo:
 *   AvoidGlobalProc0: true
 * \endcode
 *
 * If you have singletons, but do not want to assign any of them to a specific
 * proc or be exclusive on a proc, you'll need the following block in the input
 * file (where you can set the value of AvoidGlobalProc0 to true or false):
 *
 * \code {.yaml}
 * ResourceInfo:
 *   AvoidGlobalProc0: true
 *   Singletons: Auto
 * \endcode
 *
 * Otherwise, you will need to specify a block in the input file as below,
 * where you will need to specify the options for each singleton:
 *
 * \code {.yaml}
 * ResourceInfo:
 *   AvoidGlobalProc0: true
 *   Singletons:
 *     MySingleton1:
 *       Proc: 2
 *       Exclusive: true
 *     MySingleton2: Auto
 * \endcode
 *
 * where `MySingleton1` is the `pretty_type::name` of the singleton component
 * and the options for each singleton are described in
 * `Parallel::SingletonInfoHolder` (You can use `Auto` for each singleton that
 * you want to have it's proc determined automatically and be non-exclusive,
 * like `MySingleton2`).
 *
 * Several consistency checks are done during option parsing to avoid user
 * error. However, some checks can't be done during option parsing because the
 * number of nodes/procs is needed to determine if there is an inconsistency.
 * These checks are done during runtime, just before the map of singletons is
 * created.
 *
 * To automatically place singletons, we use a custom algorithm that will
 * distribute singletons evenly over the number of nodes, and evenly over the
 * procs on a node. This will help keep communication costs down by distributing
 * the workload over all of the communication cores (one communication core per
 * charm node), and ensure that our resources are being maximally utilized (i.e.
 * one core doesn't have all the singletons on it).
 *
 * Defining some terminology for singletons: `requested` means that a specific
 * processor was requested in the input file; `auto` means that the processor
 * should be chosen automatically; `exclusive` means that no other singletons or
 * array elements should be placed on this singleton's processor; `nonexclusive`
 * means that you *can* place other singletons or array elements on this
 * singleton's processor. The algorithm that distributes the singletons is as
 * follows:
 *
 * 1. Allocate all singletons that `requested` specific processors, both
 *    `exclusive` and `nonexclusive`. This is done during option parsing.
 * 2. Allocate `auto exclusive` singletons, distributing the total number of
 *    `exclusive` singletons (`auto` + `requested`) as evenly as possibly over
 *    the number of nodes. We say "as evenly as possible" because this depends
 *    on the `requested exclusive` singletons. For example, if we have 4 nodes
 *    and 5 cores per node, the number of `requested exclusive` singletons on
 *    each node is (0, 1, 4, 1), and we have 3 `auto exclusive` singletons to
 *    place, the best distribution of `exclusive` singletons we can achieve
 *    given our constraints is (2, 2, 4, 1). Clearly this is not the *most*
 *    evenly distributed the `exclusive` singletons could be. However, this *is*
 *    the most evenly distributed they could be given the starting distribution
 *    from the input file.
 * 3. Allocate `auto nonexclusive` singletons, distributing the total number of
 *    `nonexclusive` singletons (`auto` + `requested`): First, as evenly as
 *    possibly over the number of nodes. Then, on each node, distributing the
 *    singletons as evenly as possibly over the number of processors on that
 *    node. The same disclaimer about "as evenly as possibly" from the previous
 *    step applies here.
 *
 * The goal of this algorithm is to mimic, as best as possible, how a human
 * would distribute this workload. It isn't perfect, but is a significant
 * improvement over placing singletons on one proc after another starting from
 * global proc 0.
 */
template <typename Metavariables>
struct ResourceInfo {
 private:
  using singletons = detail::singleton_components<Metavariables>;

  template <typename Component>
  struct LocalTag {
    // exclusive, proc
    using type = std::pair<bool, std::optional<size_t>>;
  };
  using local_tags =
      tmpl::transform<singletons, tmpl::bind<LocalTag, tmpl::_1>>;

 public:
  struct Singletons {
    using type = Options::Auto<SingletonPack<singletons>>;
    static constexpr Options::String help = {
        "Resource options for all singletons."};
  };

  struct AvoidGlobalProc0 {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to avoid placing Array elements or singletons on global proc "
        "0."};
  };

  using options = tmpl::push_front<
      tmpl::conditional_t<tmpl::size<singletons>::value != 0,
                          tmpl::list<Singletons>, tmpl::list<>>,
      AvoidGlobalProc0>;

  static constexpr Options::String help = {
      "Resource options for a simulation. This information will be used when "
      "placing Array and Singleton parallel components on the requested "
      "resources."};

  /// The main constructor. All other constructors that take options will call
  /// this one. This constructor holds all checks able to be done during option
  /// parsing.
  ResourceInfo(const bool avoid_global_proc_0,
               const std::optional<SingletonPack<singletons>>& singleton_pack,
               const Options::Context& context = {});

  /// This constructor is used when only AvoidGlobalProc0 is specified, but no
  /// SingletonInfoHolders are specified. Calls the main constructor with an
  /// empty SingletonPack.
  ResourceInfo(const bool avoid_global_proc_0,
               const Options::Context& context = {});

  ResourceInfo() = default;
  ResourceInfo(const ResourceInfo& /*rhs*/) = default;
  ResourceInfo& operator=(const ResourceInfo& /*rhs*/) = default;
  ResourceInfo(ResourceInfo&& /*rhs*/) = default;
  ResourceInfo& operator=(ResourceInfo&& /*rhs*/) = default;
  ~ResourceInfo() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// Returns whether we should avoid placing array elements and singletons on
  /// the global zeroth proc. Default `false`.
  bool avoid_global_proc_0() const { return avoid_global_proc_0_; }

  /// Return a SingletonInfoHolder corresponding to `Component`
  template <typename Component>
  auto get_singleton_info() const;

  /// Returns a `std::unordered_set<size_t>` of processors that array components
  /// should avoid placing elements on. This should be passed to the
  /// `allocate_array` function of the array component
  const std::unordered_set<size_t>& procs_to_ignore() const;

  /// Returns a `std::set<size_t>` that has all processors available to put
  /// elements on, meaning processors that aren't ignored.
  const std::set<size_t>& procs_available_for_elements() const;

  /// Returns the proc that the singleton `Component` should be placed on.
  template <typename Component>
  size_t proc_for() const;

  /// \brief Actually builds the singleton map and allocates all the singletons.
  ///
  /// \details This could be done in the constructor, however, since we need the
  /// number of nodes to do some sanity checks, it can't. If an executable is
  /// run with the --check-options flag, we will be running on 1 proc and 1 node
  /// so some of the checks done in this function would fail. Unfortunately,
  /// that means the checks that require knowing the number of nodes now occur
  /// at runtime instead of option parsing. This is why the
  /// `singleton_map_has_been_set_` bool is necessary and why we check if this
  /// function has been called in most other member functions.
  ///
  /// To avoid a cyclic dependency between the GlobalCache and ResourceInfo, we
  /// template this function rather than explicitly use the GlobalCache because
  /// the GlobalCache depends on ResourceInfo
  ///
  /// This function should only be called once.
  template <typename Cache>
  void build_singleton_map(const Cache& cache);

 private:
  template <typename Metavars>
  friend bool operator==(const ResourceInfo<Metavars>& lhs,
                         const ResourceInfo<Metavars>& rhs);

  void singleton_map_not_built() const {
    ERROR(
        "The singleton map has not been built yet. You must call "
        "build_singleton_map() before you call this function.");
  }
  bool avoid_global_proc_0_{false};
  bool singleton_map_has_been_set_{false};
  // These are quantities that we will need for placing singletons which can be
  // determined just by option parsing
  size_t num_exclusive_singletons_{};
  size_t num_procs_to_ignore_{};
  size_t num_requested_exclusive_singletons_{};
  size_t num_requested_nonexclusive_singletons_{};
  std::unordered_multiset<size_t> requested_nonexclusive_procs_{};
  // Procs that are exclusive. These may or may not be specifically requested
  std::unordered_set<size_t> procs_to_ignore_{};
  std::set<size_t> procs_available_for_elements_{};
  // For each singleton (whether it has a SingletonInfo or not), maps whether
  // it's exclusive and what proc it is on.
  tuples::tagged_tuple_from_typelist<local_tags> singleton_map_{};
};

template <typename Metavariables>
ResourceInfo<Metavariables>::ResourceInfo(
    const bool avoid_global_proc_0,
    const std::optional<SingletonPack<singletons>>& opt_singleton_pack,
    const Options::Context& context)
    : avoid_global_proc_0_(avoid_global_proc_0) {
  if (avoid_global_proc_0_) {
    procs_to_ignore_.insert(0);
    ++num_procs_to_ignore_;
  }

  if constexpr (tmpl::size<singletons>::value > 0) {
    const auto& singleton_pack =
        opt_singleton_pack.value_or(SingletonPack<singletons>{});

    // Procs that were specifically requested. These may or may not be exclusive
    std::unordered_multiset<int> requested_procs{};

    [[maybe_unused]] const auto parse_singletons = [this, &context,
                                                    &singleton_pack,
                                                    &requested_procs](
                                                       const auto component_v) {
      using component = tmpl::type_from<decltype(component_v)>;
      auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);

      // This singleton has a SingletonInfoHolder associated with it. Get all
      // the info necessary from it
      if constexpr (tmpl::list_contains_v<singletons, component>) {
        const auto& info_holder = singleton_pack.template get<component>();
        // Assign proc. If a specific proc is requested, add it to a map. We'll
        // check that exclusive singletons have unique procs once we've gone
        // through everything once
        const auto proc = info_holder.proc();
        singleton_map.second = proc;

        if (proc.has_value()) {
          requested_procs.insert(*proc);
        }

        if (info_holder.is_exclusive()) {
          // Check that no singleton has requested to be on proc 0 while
          // AvoidGlobalProc0 is simultaneously true.
          if (avoid_global_proc_0_ and proc.has_value() and *proc == 0) {
            PARSE_ERROR(
                context,
                "A singleton has requested to be exclusively on proc 0, "
                "but the AvoidGlobalProc0 option is also set to true.");
          }

          // This singleton is exclusive so set it.
          singleton_map.first = true;
          ++num_exclusive_singletons_;
          ++num_procs_to_ignore_;
          // If it requested a specific proc, ignore it when assigning the rest
          // of the singletons
          if (proc.has_value()) {
            procs_to_ignore_.insert(static_cast<size_t>(*proc));
            ++num_requested_exclusive_singletons_;
          }
        } else {
          // This singleton is not exclusive.
          singleton_map.first = false;
          if (proc.has_value()) {
            ++num_requested_nonexclusive_singletons_;
            requested_nonexclusive_procs_.insert(static_cast<size_t>(*proc));
          }
        }
      } else {
        // This singleton doesn't have a SingletonInfoHolder so it automatically
        // isn't exclusive and gets set assigned an automatic proc.
        singleton_map.first = false;
        // nullopt is a sentinel for auto
        singleton_map.second = std::nullopt;
      }
    };

    // Create a map between each singleton, whether it is exclusive, and which
    // proc it wants to be on. Use nullopt as a sentinel for choosing the proc
    // automatically.
    tmpl::for_each<singletons>(parse_singletons);
    [[maybe_unused]] const auto sanity_checks = [this, &context,
                                                 &requested_procs](
                                                    const auto component_v) {
      using component = tmpl::type_from<decltype(component_v)>;
      auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);

      const bool exclusive = singleton_map.first;
      const auto proc = singleton_map.second;

      // Check exclusive singletons that requested to be on a specific proc
      // if any other singletons requested to be on the same proc (exclusive
      // or not)
      if (exclusive and proc.has_value() and requested_procs.count(*proc) > 1) {
        PARSE_ERROR(context,
                    "Two singletons have requested to be on proc "
                        << proc.value()
                        << ", but at least one of them has requested to be "
                           "exclusively on this proc.");
      }
    };

    // Do some inter-singleton sanity checks to avoid inconsistencies
    tmpl::for_each<singletons>(sanity_checks);
  }
}

template <typename Metavariables>
ResourceInfo<Metavariables>::ResourceInfo(const bool avoid_global_proc_0,
                                          const Options::Context& context)
    : ResourceInfo(avoid_global_proc_0, std::nullopt, context) {}

template <typename Metavariables>
void ResourceInfo<Metavariables>::pup(PUP::er& p) {
  p | avoid_global_proc_0_;
  p | singleton_map_has_been_set_;
  p | num_exclusive_singletons_;
  p | num_procs_to_ignore_;
  p | num_requested_exclusive_singletons_;
  p | num_requested_nonexclusive_singletons_;
  p | requested_nonexclusive_procs_;
  p | procs_to_ignore_;
  p | procs_available_for_elements_;
  p | singleton_map_;
}

template <typename Metavariables>
template <typename Component>
auto ResourceInfo<Metavariables>::get_singleton_info() const {
  if (not singleton_map_has_been_set_) {
    singleton_map_not_built();
  }

  const auto& singleton_map = tuples::get<LocalTag<Component>>(singleton_map_);
  return SingletonInfoHolder<Component>{
      {static_cast<int>(*singleton_map.second)}, singleton_map.first};
}

template <typename Metavariables>
const std::unordered_set<size_t>& ResourceInfo<Metavariables>::procs_to_ignore()
    const {
  if (not singleton_map_has_been_set_) {
    singleton_map_not_built();
  }
  return procs_to_ignore_;
}

template <typename Metavariables>
const std::set<size_t>&
ResourceInfo<Metavariables>::procs_available_for_elements() const {
  if (not singleton_map_has_been_set_) {
    singleton_map_not_built();
  }
  return procs_available_for_elements_;
}

template <typename Metavariables>
template <typename Component>
size_t ResourceInfo<Metavariables>::proc_for() const {
  if (not singleton_map_has_been_set_) {
    singleton_map_not_built();
  }
  return *tuples::get<LocalTag<Component>>(singleton_map_).second;
}

template <typename Metavars>
bool operator==(const ResourceInfo<Metavars>& lhs,
                const ResourceInfo<Metavars>& rhs) {
  return lhs.avoid_global_proc_0_ == rhs.avoid_global_proc_0_ and
         lhs.singleton_map_has_been_set_ == rhs.singleton_map_has_been_set_ and
         lhs.num_exclusive_singletons_ == rhs.num_exclusive_singletons_ and
         lhs.num_procs_to_ignore_ == rhs.num_procs_to_ignore_ and
         lhs.num_requested_exclusive_singletons_ ==
             rhs.num_requested_exclusive_singletons_ and
         lhs.num_requested_nonexclusive_singletons_ ==
             rhs.num_requested_nonexclusive_singletons_ and
         lhs.requested_nonexclusive_procs_ ==
             rhs.requested_nonexclusive_procs_ and
         lhs.procs_to_ignore_ == rhs.procs_to_ignore_ and
         lhs.procs_available_for_elements_ ==
             rhs.procs_available_for_elements_ and
         lhs.singleton_map_ == rhs.singleton_map_;
}

template <typename Metavars>
bool operator!=(const ResourceInfo<Metavars>& lhs,
                const ResourceInfo<Metavars>& rhs) {
  return not(lhs == rhs);
}

template <typename Metavariables>
template <typename Cache>
void ResourceInfo<Metavariables>::build_singleton_map(const Cache& cache) {
  const size_t num_procs = Parallel::number_of_procs<size_t>(cache);
  const size_t num_nodes = Parallel::number_of_nodes<size_t>(cache);

  // We don't do procs_to_ignore_.size() here because the auto singletons who
  // requested to be exclusive haven't been assigned yet so their procs haven't
  // been added to procs_to_ignore_
  if (num_procs_to_ignore_ >= num_procs) {
    ERROR(
        "The total number of cores requested is less than or equal to the "
        "number of cores that requested to be exclusive, i.e. without "
        "array elements or multiple singletons. The array elements have "
        "nowhere to be placed. Number of cores requested: "
        << num_procs << ". Number of cores that requested to be exclusive: "
        << num_procs_to_ignore_ << ".");
  }

  // Check if any singletons that requested to be on specific proc requested to
  // be on a proc beyond the last proc.
  tmpl::for_each<singletons>([this, &num_procs](const auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);
    const auto proc = singleton_map.second;

    if (proc.has_value() and *proc > num_procs - 1) {
      ERROR("Singleton " << pretty_type::name<component>()
                         << " requested to be placed on proc " << *proc
                         << ", but that proc is beyond the last proc "
                         << num_procs - 1 << ".");
    }
  });

  // At this point, all requested singletons have been allocated on their
  // desired procs. This leaves just the auto singletons left, both exclusive
  // and non-exclusive.

  // First allocate auto exclusive singletons
  // This first vector will keep track of the total number of singletons on each
  // node so we can spread them out evenly
  std::vector<size_t> singletons_on_each_node(num_nodes, 0_st);
  // This second vector keeps track of only the auto exclusive singletons on
  // each node
  std::vector<size_t> auto_exclusive_singletons_on_each_node(num_nodes, 0_st);
  // Populate requested exclusive singletons on each node with input options. We
  // couldn't have done this in the constructor because we didn't know how many
  // nodes there were or how many procs were on each node. We'll do the
  // non-exclusive ones later.
  tmpl::for_each<singletons>(
      [this, &cache, &singletons_on_each_node](const auto component_v) {
        using component = tmpl::type_from<decltype(component_v)>;
        auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);
        const bool exclusive = singleton_map.first;
        const auto proc = singleton_map.second;

        if (exclusive and proc.has_value()) {
          ++singletons_on_each_node[Parallel::node_of<size_t>(*proc, cache)];
        }
      });

  size_t remaining_auto_exclusive_singletons =
      num_exclusive_singletons_ - num_requested_exclusive_singletons_;
  // Start with the min number of singletons on a node as our baseline. Then,
  // while we still have auto exclusive singletons to place, we loop over all
  // nodes and place singletons on nodes with this minimum number. Once all
  // nodes have at least this minimum number, we increment the minimum number
  // and loop over the nodes again
  size_t min_num_singletons_on_a_node = *std::min_element(
      singletons_on_each_node.begin(), singletons_on_each_node.end());
  while (remaining_auto_exclusive_singletons > 0) {
    for (size_t i = 0; i < num_nodes; i++) {
      // If this node has more than the minimum number of singletons on it, skip
      // it for now
      if (singletons_on_each_node[i] > min_num_singletons_on_a_node) {
        continue;
      }
      // Since nodes can have different number of procs, we check that we
      // haven't exhausted the number of procs on this node. This check is ok
      // right now because we haven't included any nonexclusive singletons in
      // singletons_on_each_node yet.
      if (not(singletons_on_each_node[i] <
              Parallel::procs_on_node<size_t>(i, cache))) {
        continue;
      }

      ++singletons_on_each_node[i];
      ++auto_exclusive_singletons_on_each_node[i];
      --remaining_auto_exclusive_singletons;

      // We need to break out of both loops here. Use a goto.
      if (remaining_auto_exclusive_singletons == 0) {
        goto break_auto_exclusive_loops;
      }
    }  // for (size_t i = 0; i < num_nodes; i++)

    ++min_num_singletons_on_a_node;
  }  // while (remaining_auto_exclusive_singletons > 0)
break_auto_exclusive_loops:

  ASSERT(remaining_auto_exclusive_singletons == 0,
         "Not all exclusive singletons have been allocated. The remaining "
         "number of singletons to be allocated is "
             << remaining_auto_exclusive_singletons << ".");

  // Actually allocate the auto exclusive singletons
  size_t current_node = 0;
  tmpl::for_each<singletons>([this, &cache, &current_node,
                              &auto_exclusive_singletons_on_each_node](
                                 const auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);
    const bool exclusive = singleton_map.first;
    const auto int_proc = singleton_map.second;

    // Only allocating auto exclusive at the moment
    if (exclusive and not int_proc.has_value()) {
      while (auto_exclusive_singletons_on_each_node[current_node] == 0) {
        ++current_node;
      }

      size_t proc = Parallel::first_proc_on_node<size_t>(current_node, cache);
      // Don't place two exclusive singletons on the same proc, but also if
      // a singleton requested a specific proc, whether or not it is
      // exclusive, we can't place an exclusive singleton on that proc. That
      // defeats the whole purpose of requesting the specific proc...
      while (procs_to_ignore_.find(proc) != procs_to_ignore_.end() or
             requested_nonexclusive_procs_.count(proc) > 0) {
        ++proc;
      }

      singleton_map.second = proc;
      procs_to_ignore_.insert(proc);

      --auto_exclusive_singletons_on_each_node[current_node];
    }
  });

  ASSERT(alg::accumulate(auto_exclusive_singletons_on_each_node, 0_st) == 0,
         "Not all auto exclusive singletons have been allocated. The remaining "
         "number of auto exclusive singletons to be allocated is "
             << alg::accumulate(auto_exclusive_singletons_on_each_node, 0_st));

  // procs_to_ignore_ is now complete. Now construct
  // procs_available_for_elements_
  for (size_t i = 0; i < num_procs; i++) {
    if (procs_to_ignore_.find(i) == procs_to_ignore_.end()) {
      procs_available_for_elements_.insert(i);
    }
  }

  // At this point, all auto exclusive singletons have been allocated. Now the
  // only singletons left are auto non-exclusive. We use vectors of
  // std::optional<size_t> here as sentinels for procs which should be avoided.
  // A nullopt means that the proc shouldn't have auto nonexclusive singletons
  // on it. When we have lots of cores to run on (hundreds of thousands or even
  // millions), these vectors will take up a non-negligible amount of memory.
  // However, we only need to do this once an executable at the very beginning
  // so it shouldn't really matter
  std::vector<std::optional<size_t>> nonexclusive_singletons_on_each_proc(
      num_procs, std::optional<size_t>(0_st));
  // This vector has the default be nullopt rather than 0, because this will
  // only be used when we actually place singletons. We only care which procs
  // have singletons, which will usually be a small subset of the total procs.
  std::vector<std::optional<size_t>> auto_nonexclusive_singletons_on_each_proc(
      num_procs, std::nullopt);
  for (const size_t proc : procs_to_ignore_) {
    nonexclusive_singletons_on_each_proc[proc] = std::nullopt;
  }
  // Now we add in the requested nonexclusive to the total number of singletons
  // per node
  for (const auto& proc : requested_nonexclusive_procs_) {
    ++*nonexclusive_singletons_on_each_proc[proc];
    ++singletons_on_each_node[Parallel::node_of<size_t>(proc, cache)];
  }

  size_t remaining_auto_nonexclusive_singletons =
      tmpl::size<singletons>::value - num_exclusive_singletons_ -
      num_requested_nonexclusive_singletons_;

  // This serves the same purpose as before
  min_num_singletons_on_a_node = *std::min_element(
      singletons_on_each_node.begin(), singletons_on_each_node.end());
  while (remaining_auto_nonexclusive_singletons > 0) {
    for (size_t i = 0; i < num_nodes; i++) {
      const int first_proc = Parallel::first_proc_on_node<int>(i, cache);
      const int procs_on_node = Parallel::procs_on_node<int>(i, cache);
      const int first_proc_next_node = first_proc + procs_on_node;

      auto first_proc_iter =
          std::next(nonexclusive_singletons_on_each_proc.begin(), first_proc);
      auto first_proc_next_node_iter = std::next(
          nonexclusive_singletons_on_each_proc.begin(), first_proc_next_node);

      // Get the proc on this node with the minimum number of singletons. This
      // serves the same purpose as min_num_singletons_on_a_node except now for
      // procs on a specific node
      auto& min_num_singletons_on_a_proc_opt =
          *std::min_element(first_proc_iter, first_proc_next_node_iter,
                            [](const auto& a, const auto& b) {
                              if (a.has_value() and b.has_value()) {
                                return a.value() < b.value();
                              } else {
                                return a.has_value();
                              }
                            });

      // Check if this node can accommodate more singletons. Do two checks:
      // 1. This node doesn't have more than the minimum number of singletons
      // 2. That this node isn't filled up with exclusive singletons (nullopt =
      //    all procs on this node are taken)
      if (singletons_on_each_node[i] > min_num_singletons_on_a_node or
          not min_num_singletons_on_a_proc_opt.has_value()) {
        continue;
      }

      // At this point, we have guaranteed that this node should have an auto
      // nonexclusive singleton on it somewhere. Now determine where
      size_t min_num_singletons_on_a_proc =
          min_num_singletons_on_a_proc_opt.value();

      // Find the first available proc on this node. Check that
      // 1. This proc is available (i.e. no exclusive singletons on it)
      // 2. This proc has the minimum number of singletons on it for this node
      //    so we distribute the singletons evenly over all the procs on this
      //    node
      auto proc_iter =
          std::find_if(first_proc_iter, first_proc_next_node_iter,
                       [&min_num_singletons_on_a_proc](const auto& proc_opt) {
                         return proc_opt.has_value() and
                                *proc_opt == min_num_singletons_on_a_proc;
                       });

      // Get the index of the overall vector. We need this because we're going
      // to be indexing two separate vectors, otherwise we could have just used
      // the value of the iterator
      const size_t proc = static_cast<size_t>(std::distance(
          nonexclusive_singletons_on_each_proc.begin(), proc_iter));

      // Increment things
      ++*nonexclusive_singletons_on_each_proc[proc];
      ++singletons_on_each_node[i];
      if (auto_nonexclusive_singletons_on_each_proc[proc].has_value()) {
        ++*auto_nonexclusive_singletons_on_each_proc[proc];
      } else {
        auto_nonexclusive_singletons_on_each_proc[proc] = 1;
      }
      --remaining_auto_nonexclusive_singletons;

      // We need to break out of both loops here. Use a goto.
      if (remaining_auto_nonexclusive_singletons == 0) {
        goto break_auto_nonexclusive_loops;
      }
    }  // for (size_t i = 0; i < num_nodes; i++)

    ++min_num_singletons_on_a_node;
  }  // while (remaining_auto_nonexclusive_singletons > 0)
break_auto_nonexclusive_loops:

  ASSERT(remaining_auto_nonexclusive_singletons == 0,
         "Not all nonexclusive singletons have been allocated. The remaining "
         "number of singletons to be allocated is "
             << remaining_auto_nonexclusive_singletons << ".");

  // Actually allocate the auto nonexclusive singletons
  std::stringstream ss;
  ss << "\nAllocating Singletons:\n";
  size_t current_proc = 0;
  tmpl::for_each<singletons>([this, &current_proc, &cache, &ss,
                              &auto_nonexclusive_singletons_on_each_proc](
                                 const auto component_v) {
    using component = tmpl::type_from<decltype(component_v)>;
    auto& singleton_map = tuples::get<LocalTag<component>>(singleton_map_);
    const auto proc_opt = singleton_map.second;

    // At this point, the only singletons that have this are nonexclusive
    if (not proc_opt.has_value()) {
      while (not auto_nonexclusive_singletons_on_each_proc[current_proc]
                     .has_value()) {
        ++current_proc;
      }

      singleton_map.second = current_proc;

      --*auto_nonexclusive_singletons_on_each_proc[current_proc];
      // Indicate that there are no more singletons to be placed on this proc
      if (*auto_nonexclusive_singletons_on_each_proc[current_proc] == 0) {
        auto_nonexclusive_singletons_on_each_proc[current_proc] = std::nullopt;
      }
    }

    // Print some diagnostic info to stdout for each singleton. This can aid in
    // debugging.
    ss << pretty_type::name<component>();
    ss << " on node " << Parallel::node_of<int>(*singleton_map.second, cache);
    ss << ", global proc " << *singleton_map.second;
    ss << ", exclusive = " << std::boolalpha << singleton_map.first << "\n";
  });

  ss << "\n";
  Parallel::printf("%s", ss.str());

  // Now that everything has been set, signal that we don't have to do
  // this again.
  singleton_map_has_been_set_ = true;
}
}  // namespace Parallel
