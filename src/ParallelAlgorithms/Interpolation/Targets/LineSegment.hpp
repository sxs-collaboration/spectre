// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
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
template <typename TemporalId>
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
    static constexpr Options::String help = {"Beginning endpoint"};
  };
  struct End {
    using type = std::array<double, VolumeDim>;
    static constexpr Options::String help = {"Ending endpoint"};
  };
  struct NumberOfPoints {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of points including endpoints"};
    static type lower_bound() { return 2; }
  };
  using options = tmpl::list<Begin, End, NumberOfPoints>;
  static constexpr Options::String help = {
      "A line segment extending from Begin to End, containing NumberOfPoints"
      " uniformly-spaced points including the endpoints."};

  LineSegment(std::array<double, VolumeDim> begin_in,
              std::array<double, VolumeDim> end_in, size_t number_of_points_in);

  LineSegment() = default;
  LineSegment(const LineSegment& /*rhs*/) = delete;
  LineSegment& operator=(const LineSegment& /*rhs*/) = default;
  LineSegment(LineSegment&& /*rhs*/) = default;
  LineSegment& operator=(LineSegment&& /*rhs*/) = default;
  ~LineSegment() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p);  // NOLINT

  std::array<double, VolumeDim> begin{};
  std::array<double, VolumeDim> end{};
  size_t number_of_points{};
};

template <size_t VolumeDim>
bool operator==(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs);
template <size_t VolumeDim>
bool operator!=(const LineSegment<VolumeDim>& lhs,
                const LineSegment<VolumeDim>& rhs);

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment {
  using type = OptionHolders::LineSegment<VolumeDim>;
  static constexpr Options::String help{
      "Options for interpolation onto line segment."};
  static std::string name() { return Options::name<InterpolationTargetTag>(); }
  using group = InterpolationTargets;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment : db::SimpleTag {
  using type = OptionHolders::LineSegment<VolumeDim>;
  using option_tags =
      tmpl::list<OptionTags::LineSegment<InterpolationTargetTag, VolumeDim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Computes points on a line segment.
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, size_t VolumeDim>
struct LineSegment {
  using const_global_cache_tags =
      tmpl::list<Tags::LineSegment<InterpolationTargetTag, VolumeDim>>;
  using is_sequential = std::false_type;
  using frame = Frame::Inertial;

  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, VolumeDim, Frame::Inertial> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) {
    const auto& options =
        get<Tags::LineSegment<InterpolationTargetTag, VolumeDim>>(box);

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
    return target_points;
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, VolumeDim, Frame::Inertial> points(
      const db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& meta,
      const TemporalId& /*temporal_id*/) {
    return points(box, meta);
  }
};

}  // namespace TargetPoints
}  // namespace intrp
