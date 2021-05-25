// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
/// A list of specified points to interpolate to.
///
/// \note Input coordinates are interpreted in `Frame::Inertial`
template <size_t VolumeDim>
struct SpecifiedPoints {
  struct Points {
    using type = std::vector<std::array<double, VolumeDim>>;
    static constexpr Options::String help = {"Coordinates of each point"};
  };
  using options = tmpl::list<Points>;
  static constexpr Options::String help = {"A list of specified points"};

  explicit SpecifiedPoints(
      std::vector<std::array<double, VolumeDim>> points_in) noexcept;

  SpecifiedPoints() = default;
  SpecifiedPoints(const SpecifiedPoints& /*rhs*/) = delete;
  SpecifiedPoints& operator=(const SpecifiedPoints& /*rhs*/) = default;
  SpecifiedPoints(SpecifiedPoints&& /*rhs*/) noexcept = default;
  SpecifiedPoints& operator=(SpecifiedPoints&& /*rhs*/) noexcept = default;
  ~SpecifiedPoints() = default;

  // clang-tidy non-const reference pointer.
  void pup(PUP::er& p) noexcept;  // NOLINT

  std::vector<std::array<double, VolumeDim>> points{};
};

template <size_t VolumeDim>
bool operator==(const SpecifiedPoints<VolumeDim>& lhs,
                const SpecifiedPoints<VolumeDim>& rhs) noexcept;
template <size_t VolumeDim>
bool operator!=(const SpecifiedPoints<VolumeDim>& lhs,
                const SpecifiedPoints<VolumeDim>& rhs) noexcept;

}  // namespace OptionHolders

namespace OptionTags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct SpecifiedPoints {
  using type = OptionHolders::SpecifiedPoints<VolumeDim>;
  static constexpr Options::String help{
      "Options for interpolation onto a specified list of points."};
  static std::string name() noexcept {
    return Options::name<InterpolationTargetTag>();
  }
  using group = InterpolationTargets;
};
}  // namespace OptionTags

namespace Tags {
template <typename InterpolationTargetTag, size_t VolumeDim>
struct SpecifiedPoints : db::SimpleTag {
  using type = OptionHolders::SpecifiedPoints<VolumeDim>;
  using option_tags = tmpl::list<
      OptionTags::SpecifiedPoints<InterpolationTargetTag, VolumeDim>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) noexcept {
    return option;
  }
};
}  // namespace Tags

namespace TargetPoints {
/// \brief Returns list of points as specified in input file.
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag, size_t VolumeDim>
struct SpecifiedPoints {
  using const_global_cache_tags =
      tmpl::list<Tags::SpecifiedPoints<InterpolationTargetTag, VolumeDim>>;
  using is_sequential = std::false_type;
  using frame = Frame::Inertial;

  template <typename Metavariables, typename DbTags>
  static tnsr::I<DataVector, VolumeDim, Frame::Inertial> points(
      const db::DataBox<DbTags>& box,
      const tmpl::type_<Metavariables>& /*meta*/) noexcept {
    const auto& options =
        get<Tags::SpecifiedPoints<InterpolationTargetTag, VolumeDim>>(box);
    tnsr::I<DataVector, VolumeDim, Frame::Inertial> target_points(
        options.points.size());
    for (size_t d = 0; d < VolumeDim; ++d) {
      for (size_t i = 0; i < options.points.size(); ++i) {
        target_points.get(d)[i] = gsl::at(options.points[i], d);
      }
    }
    return target_points;
  }

  template <typename Metavariables, typename DbTags, typename TemporalId>
  static tnsr::I<DataVector, VolumeDim, Frame::Inertial> points(
      const db::DataBox<DbTags>& box, const tmpl::type_<Metavariables>& meta,
      const TemporalId& /*temporal_id*/) noexcept {
    return points(box, meta);
  }
};

}  // namespace TargetPoints
}  // namespace intrp
