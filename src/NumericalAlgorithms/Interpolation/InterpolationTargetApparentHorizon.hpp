// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Informer/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace intrp {
namespace Tags {
template <typename Metavariables>
struct TemporalIds;
}  // namespace Tags
}  // namespace intrp
namespace OptionTags {
struct Verbosity;
}  // namespace OptionTags
namespace Tags {
struct Verbosity;
}  // namespace Tags
/// \endcond

namespace intrp {

namespace OptionHolders {
/// Options for finding an apparent horizon.
template <typename Frame>
struct ApparentHorizon {
  /// See Strahlkorper for suboptions.
  struct InitialGuess {
    static constexpr OptionString help = {"Initial guess"};
    using type = Strahlkorper<Frame>;
  };
  /// See ::FastFlow for suboptions.
  struct FastFlow {
    static constexpr OptionString help = {"FastFlow options"};
    using type = ::FastFlow;
  };
  using options = tmpl::list<InitialGuess, FastFlow, ::OptionTags::Verbosity>;
  static constexpr OptionString help = {
      "Provide an initial guess for the apparent horizon surface\n"
      "(Strahlkorper) and apparent-horizon-finding-algorithm (FastFlow)\n"
      "options."};

  ApparentHorizon(Strahlkorper<Frame> initial_guess_in, ::FastFlow fast_flow_in,
                  Verbosity verbosity_in) noexcept;

  ApparentHorizon() = default;
  ApparentHorizon(const ApparentHorizon& /*rhs*/) = delete;
  ApparentHorizon& operator=(const ApparentHorizon& /*rhs*/) = delete;
  ApparentHorizon(ApparentHorizon&& /*rhs*/) noexcept = default;
  ApparentHorizon& operator=(ApparentHorizon&& /*rhs*/) noexcept = default;
  ~ApparentHorizon() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept;  // NOLINT

  Strahlkorper<Frame> initial_guess{};
  ::FastFlow fast_flow{};
  Verbosity verbosity{Verbosity::Quiet};
};

template <typename Frame>
bool operator==(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs) noexcept;
template <typename Frame>
bool operator!=(const ApparentHorizon<Frame>& lhs,
                const ApparentHorizon<Frame>& rhs) noexcept;

}  // namespace OptionHolders

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sends points on a trial apparent horizon to an `Interpolator`.
///
/// This differs from `KerrHorizon` in the following ways:
/// - It supplies points on a prolonged Strahlkorper, at a higher resolution
///   than the Strahlkorper in the DataBox, as needed for horizon finding.
/// - It uses a `FastFlow` in the DataBox.
/// - It has different options (including those for `FastFlow`).
///
/// Uses:
/// - DataBox:
///   - `::Tags::Domain<VolumeDim, Frame>`
///   - `::ah::Tags::FastFlow`
///   - `StrahlkorperTags::CartesianCoords<Frame>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
///
/// This Action also has an initialize function that adds to the DataBox:
/// - `StrahlkorperTags::items_tags<Frame>`
/// - `StrahlkorperTags::compute_items_tags<Frame>`
/// - `::ah::Tags::FastFlow`
/// - `::Tags::Verbosity`
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, typename Frame>
struct ApparentHorizon {
  using options_type = OptionHolders::ApparentHorizon<Frame>;
  using const_global_cache_tags = tmpl::list<InterpolationTargetTag>;
  using initialization_tags =
      tmpl::append<StrahlkorperTags::items_tags<Frame>,
                   tmpl::list<::ah::Tags::FastFlow, ::Tags::Verbosity>,
                   StrahlkorperTags::compute_items_tags<Frame>>;
  template <typename DbTags, typename Metavariables>
  static auto initialize(
      db::DataBox<DbTags>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const auto& options = Parallel::get<InterpolationTargetTag>(cache);

    // Put Strahlkorper and its ComputeItems, FastFlow,
    // and verbosity into a new DataBox.
    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<
            tmpl::push_back<StrahlkorperTags::items_tags<Frame>,
                            ::ah::Tags::FastFlow, ::Tags::Verbosity>>,
        db::AddComputeTags<StrahlkorperTags::compute_items_tags<Frame>>>(
        std::move(box), options.initial_guess, options.fast_flow,
        options.verbosity);
  }
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTags, typename Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    const auto& fast_flow = db::get<::ah::Tags::FastFlow>(box);
    const auto& strahlkorper =
        db::get<StrahlkorperTags::Strahlkorper<Frame>>(box);

    const size_t L_mesh = fast_flow.current_l_mesh(strahlkorper);
    const auto prolonged_strahlkorper =
        Strahlkorper<Frame>(L_mesh, L_mesh, strahlkorper);

    const auto prolonged_coords =
        StrahlkorperTags::CartesianCoords<Frame>::function(
            prolonged_strahlkorper,
            StrahlkorperTags::Radius<Frame>::function(prolonged_strahlkorper),
            StrahlkorperTags::Rhat<Frame>::function(
                StrahlkorperTags::ThetaPhi<Frame>::function(
                    prolonged_strahlkorper)));

    // In the future, when we add support for multiple Frames,
    // the code that transforms coordinates from the Strahlkorper Frame
    // to `Metavariables::domain_frame` will go here.  That transformation
    // may depend on `temporal_id`.

    send_points_to_interpolator<InterpolationTargetTag>(
        box, cache, prolonged_coords, temporal_id);
  }
};

}  // namespace Actions
}  // namespace intrp
