// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

namespace OptionHolders {
/// A line segment extending from `Begin` to `End`,
/// containing `NumberOfPoints` uniformly-spaced points including the endpoints.
template <size_t VolumeDim>
struct LineSegment {
  struct Begin {
    using type = std::array<double, VolumeDim>;
    static constexpr OptionString help = {"Beginning endpoint"};
  };
  struct End {
    using type = std::array<double, VolumeDim>;
    static constexpr OptionString help = {"Ending endpoint"};
  };
  struct NumberOfPoints {
    using type = size_t;
    static constexpr OptionString help = {
        "Number of points including endpoints"};
    static type lower_bound() { return 2; }
  };
  using options = tmpl::list<Begin, End, NumberOfPoints>;
  static constexpr OptionString help = {
      "A line segment extending from Begin to End, containing NumberOfPoints"
      " uniformly-spaced points including the endpoints."};

  LineSegment(std::array<double, VolumeDim> begin_in,
              std::array<double, VolumeDim> end_in,
              size_t number_of_points_in)
      : begin(std::move(begin_in)),  // NOLINT
        end(std::move(end_in)),      // NOLINT
        number_of_points(number_of_points_in) {}
  // above NOLINT for std::move of trivially copyable type.

  LineSegment() = default;
  LineSegment(const LineSegment& /*rhs*/) = default;
  LineSegment& operator=(const LineSegment& /*rhs*/) = default;
  LineSegment(LineSegment&& /*rhs*/) noexcept = default;
  LineSegment& operator=(LineSegment&& /*rhs*/) noexcept = default;
  ~LineSegment() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | begin;
    p | end;
    p | number_of_points;
  }

  std::array<double, VolumeDim> begin{};
  std::array<double, VolumeDim> end{};
  size_t number_of_points{};
};
}  // namespace OptionHolders

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sends points on a line segment to an `Interpolator`.
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
template <typename InterpolationTargetTag, size_t VolumeDim, typename Frame>
struct LineSegment {
  using options_type = OptionHolders::LineSegment<VolumeDim>;
  using const_global_cache_tags = tmpl::list<InterpolationTargetTag>;
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
      const typename Metavariables::temporal_id& temporal_id) noexcept {
    const auto& options = Parallel::get<InterpolationTargetTag>(cache);

    // Fill points on a line segment
    const double fractional_distance = 1.0 / (options.number_of_points - 1);
    tnsr::I<DataVector, VolumeDim, Frame> target_points(
        options.number_of_points);
    for (size_t n = 0; n < options.number_of_points; ++n) {
      for (size_t d = 0; d < VolumeDim; ++d) {
        target_points.get(d)[n] =
            gsl::at(options.begin, d) +
            n * fractional_distance *
                (gsl::at(options.end, d) - gsl::at(options.begin, d));
      }
    }

    send_points_to_interpolator<InterpolationTargetTag>(
        box, cache, target_points, temporal_id);
  }
};

}  // namespace Actions
}  // namespace intrp
