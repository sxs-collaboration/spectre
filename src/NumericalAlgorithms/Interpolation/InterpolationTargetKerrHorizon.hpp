// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
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
/// \endcond

namespace intrp {

namespace OptionHolders {
/// A surface that conforms to the horizon of a Kerr black hole in
/// Kerr-Schild coordinates.
struct KerrHorizon {
  struct Lmax {
    using type = size_t;
    static constexpr OptionString help = {
        "KerrHorizon is expanded in Ylms up to l=Lmax"};
  };
  struct Center {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"Center of black hole"};
  };
  struct Mass {
    using type = double;
    static constexpr OptionString help = {"Mass of black hole"};
  };
  struct DimensionlessSpin {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"Dimensionless spin of black hole"};
  };
  using options = tmpl::list<Lmax, Center, Mass, DimensionlessSpin>;
  static constexpr OptionString help = {
      "A Strahlkorper conforming to the horizon (in Kerr-Schild coordinates)"
      " of a Kerr black hole with a specified center, mass, and spin."};

  KerrHorizon(size_t l_max_in, std::array<double, 3> center_in, double mass_in,
              std::array<double, 3> dimensionless_spin_in,
              const OptionContext& context = {});

  KerrHorizon() = default;
  KerrHorizon(const KerrHorizon& /*rhs*/) = delete;
  KerrHorizon& operator=(const KerrHorizon& /*rhs*/) = delete;
  KerrHorizon(KerrHorizon&& /*rhs*/) noexcept = default;
  KerrHorizon& operator=(KerrHorizon&& /*rhs*/) noexcept = default;
  ~KerrHorizon() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept;  // NOLINT

  size_t l_max{};
  std::array<double, 3> center{};
  double mass{};
  std::array<double, 3> dimensionless_spin{};
};

bool operator==(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept;
bool operator!=(const KerrHorizon& lhs, const KerrHorizon& rhs) noexcept;

}  // namespace OptionHolders

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sends points on a Kerr horizon to an `Interpolator`.
///
/// The points are such that they conform to the horizon of a Kerr
/// black hole (in Kerr-Schild coordinates) with given center, mass,
/// and dimensionless spin, as specified in the options.
///
/// \note The returned points are not actually on the horizon;
/// instead, they are collocation points of a Strahlkorper whose
/// spectral representation matches the horizon shape up to order
/// Lmax, where Lmax is the spherical-harmonic order specified in the
/// options.  As Lmax increases, the returned points converge to the
/// horizon.
///
/// Uses:
/// - DataBox:
///   - `::Tags::Domain<VolumeDim, Frame>`
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
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, typename Frame>
struct KerrHorizon {
  using options_type = OptionHolders::KerrHorizon;
  using const_global_cache_tags = tmpl::list<InterpolationTargetTag>;
  using initialization_tags =
      tmpl::append<StrahlkorperTags::items_tags<Frame>,
                   StrahlkorperTags::compute_items_tags<Frame>>;
  template <typename DbTags, typename Metavariables>
  static auto initialize(
      db::DataBox<DbTags>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const auto& options = Parallel::get<InterpolationTargetTag>(cache);

    // Make a Strahlkorper with the correct shape.
    ::Strahlkorper<Frame> strahlkorper(
        options.l_max, options.l_max,
        get(gr::Solutions::kerr_horizon_radius(
            ::YlmSpherepack(options.l_max, options.l_max).theta_phi_points(),
            options.mass, options.dimensionless_spin)),
        options.center);

    // Put Strahlkorper and its ComputeItems into a new DataBox.
    return db::create_from<
        db::RemoveTags<>,
        db::AddSimpleTags<StrahlkorperTags::items_tags<Frame>>,
        db::AddComputeTags<StrahlkorperTags::compute_items_tags<Frame>>>(
        std::move(box), std::move(strahlkorper));
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
    // In the future, when we add support for multiple Frames,
    // the code that transforms coordinates from the Strahlkorper Frame
    // to `Metavariables::domain_frame` will go here.  That transformation
    // may depend on `temporal_id`.
    send_points_to_interpolator<InterpolationTargetTag>(
        box, cache,
        db::get<StrahlkorperTags::CartesianCoords<
            typename Metavariables::domain_frame>>(box),
        temporal_id);
  }
};

}  // namespace Actions
}  // namespace intrp
