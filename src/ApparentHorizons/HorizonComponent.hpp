// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/HorizonComponentActions.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/Tags.hpp"
#include "Informer/Tags.hpp"
#include "Informer/Verbosity.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup SurfacesGroup
/// Holds objects used by horizon finders.
namespace ah {

// This struct groups together options that can be specified
// for a single horizon, so that in the input file they can
// be specified under headings such as "AhA", "AhB", etc.
namespace Finder_detail {
template <typename Frame>
struct OptionHolder {
  struct Verbosity {
    using type = ::Verbosity;
    static constexpr OptionString help = {"Verbosity"};
    static type default_value() { return ::Verbosity::Quiet; }
  };
  struct FastFlow {
    using type = ::FastFlow;
    static constexpr OptionString help = {"FastFlow"};
  };
  struct InitialGuess {
    using type = Strahlkorper<Frame>;
    static constexpr OptionString help = {"Strahlkorper for initial guess"};
  };
  using options = tmpl::list<Verbosity, FastFlow, InitialGuess>;
  static constexpr OptionString help = {"Options for horizon finder"};

  OptionHolder(::Verbosity verbosity_in, ::FastFlow&& fast_flow_in,
               Strahlkorper<Frame>&& initial_guess_in)
      : verbosity(verbosity_in),
        // clang-tidy: move of trivially copyable type
        fast_flow(std::move(fast_flow_in)),  // NOLINT
        initial_guess(std::move(initial_guess_in)) {}

  OptionHolder() = default;
  OptionHolder(const OptionHolder& /*rhs*/) = default;
  OptionHolder& operator=(const OptionHolder& /*rhs*/) = default;
  OptionHolder(OptionHolder&& /*rhs*/) noexcept = default;
  OptionHolder& operator=(OptionHolder&& /*rhs*/) noexcept = default;
  ~OptionHolder() = default;

  ::Verbosity verbosity{::Verbosity::Quiet};
  ::FastFlow fast_flow{};
  Strahlkorper<Frame> initial_guess{};
};
}  // namespace Finder_detail

/// Tags for options.
namespace OptionTags {
// I haven't found a way to avoid having three different structs here.
// (note that the option name in the input file is the struct name,
//  so something like template<typename AhTag> struct AhOptions{...};
//  doesn't work because it will look for 'AhOptions' in the input file).
template <typename Frame>
struct AhA {
  using type = Finder_detail::OptionHolder<Frame>;
  static constexpr OptionString help = {"Options for AhA"};
};
template <typename Frame>
struct AhB {
  using type = Finder_detail::OptionHolder<Frame>;
  static constexpr OptionString help = {"Options for AhB"};
};
template <typename Frame>
struct AhC {
  using type = Finder_detail::OptionHolder<Frame>;
  static constexpr OptionString help = {"Options for AhC"};
};
}  // namespace OptionTags

/// Singleton component that holds a horizon.
///
/// \details
/// A ah::Finder is responsible for finding a horizon and
/// communicating with DataInterpolators to do so.  There are
/// only a small number of ah::Finders, one for each horizon
/// (e.g. for BBH inspirals there should be exactly three of them, which are
/// often labeled AhA, AhB, and AhC; AhC is the common horizon that pops
/// up at merger).
///
/// The AhTag is anything that has the following functions and type aliases:
///   - a static function 'label()' returning a std::string or a const char *.
///   - a type alias 'frame' indicating a frame from namespace ::Frame.
///   - a type alias 'option_tag' containing something in Horizon::OptionTags.
///   - a type alias 'convergence_hook' to a struct with a static function
///     void apply(const Strahlkorper<AhTag::frame>&, const Time&,
///                const Parallel::ConstGlobalCache<Metavariables>&) noexcept;
///     that Finder will call when it converges.
template <class Metavariables, typename AhTag>
struct Finder {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename Actions::Finder::Initialize<AhTag>::return_tag_list>;
  using options =
      tmpl::list<typename AhTag::option_tag,
                 ::OptionTags::DomainCreator<3, typename AhTag::frame>>;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
      Finder_detail::OptionHolder<typename AhTag::frame>&& option_holder,
      std::unique_ptr<DomainCreator<3, Frame::Inertial>>
          domain_creator) noexcept;
  static void execute_next_phase(
      typename metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<metavariables>&
      /*global_cache*/) noexcept {};
};

template <class Metavariables, typename AhTag>
void Finder<Metavariables, AhTag>::initialize(
    Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
    Finder_detail::OptionHolder<typename AhTag::frame>&& option_holder,
    std::unique_ptr<DomainCreator<3, Frame::Inertial>>
        domain_creator) noexcept {
  auto& my_proxy =
      Parallel::get_parallel_component<Finder>(*(global_cache.ckLocalBranch()));
  auto domain = domain_creator->create_domain();
  Parallel::simple_action<Actions::Finder::Initialize<AhTag>>(
      my_proxy, std::move(option_holder.verbosity),
      std::move(option_holder.fast_flow),
      std::move(option_holder.initial_guess), std::move(domain));
}
}  // namespace ah
