// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/SendPointsToInterpolator.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
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
/// A line segment extending from `Begin` to `End`,
/// containing `NumberOfPoints` uniformly-spaced points including the endpoints.
///
/// \note Input coordinates are interpreted in `Frame::Inertial`
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
    static type lower_bound() noexcept { return 2; }
  };
  using options = tmpl::list<Begin, End, NumberOfPoints>;
  static constexpr OptionString help = {
      "A line segment extending from Begin to End, containing NumberOfPoints"
      " uniformly-spaced points including the endpoints."};

  LineSegment(std::array<double, VolumeDim> begin_in,
              std::array<double, VolumeDim> end_in,
              size_t number_of_points_in) noexcept;

  LineSegment() = default;
  LineSegment(const LineSegment& /*rhs*/) = delete;
  LineSegment& operator=(const LineSegment& /*rhs*/) = default;
  LineSegment(LineSegment&& /*rhs*/) noexcept = default;
  LineSegment& operator=(LineSegment&& /*rhs*/) noexcept = default;
  ~LineSegment() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept;  // NOLINT

  std::array<double, VolumeDim> begin{};
  std::array<double, VolumeDim> end{};
  size_t number_of_points{};
};

template <size_t VolumeDim>
bool operator==(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) noexcept;
template <size_t VolumeDim>
bool operator!=(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs) noexcept;

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment {
  using type = OptionHolders::LineSegment<VolumeDim>;
  static constexpr OptionString help{
      "Options for interpolation onto line segment."};
  static std::string name() noexcept {
    return option_name<InterpolationTargetTag>();
  }
  using group = InterpolationTargets;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment : db::SimpleTag {
  using type = OptionHolders::LineSegment<VolumeDim>;
  using option_tags =
      tmpl::list<OptionTags::LineSegment<InterpolationTargetTag, VolumeDim>>;

  template <typename Metavariables>
  static type create_from_options(const type& option) noexcept {
    return option;
  }
};
}  // namespace Tags

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Sends points on a line segment to an `Interpolator`.
///
/// Uses:
/// - DataBox:
///   - `domain::Tags::Domain<3>`
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
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment {
  using const_global_cache_tags =
      tmpl::list<Tags::LineSegment<InterpolationTargetTag, VolumeDim>>;
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex,
            Requires<tmpl::list_contains_v<
                DbTags, Tags::TemporalIds<Metavariables>>> = nullptr>
  static void apply(
      db::DataBox<DbTags>& box,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const typename Metavariables::temporal_id::type& temporal_id) noexcept {
    const auto& options =
        Parallel::get<Tags::LineSegment<InterpolationTargetTag, VolumeDim>>(
            cache);

    // Fill points on a line segment
    const double fractional_distance = 1.0 / (options.number_of_points - 1);
    tnsr::I<DataVector, VolumeDim, Frame::Inertial> target_points(
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
