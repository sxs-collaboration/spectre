// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Options.hpp"

namespace Cce {
/// \cond
class Interpolator;
/// \endcond

namespace OptionTags {

/// %Option group
struct Cce {
  static constexpr OptionString help = {"Options for the Cce evolution system"};
};

/// %Option group
struct Filtering {
  static constexpr OptionString help = {"Options for the filtering in Cce"};
  using group = Cce;
};

struct LMax {
  using type = size_t;
  static constexpr OptionString help{
      "Maximum l value for spin-weighted spherical harmonics"};
  using group = Cce;
};

struct FilterLMax {
  using type = size_t;
  static constexpr OptionString help{"l mode cutoff for angular filtering"};
  using group = Filtering;
};

struct RadialFilterAlpha {
  using type = double;
  static constexpr OptionString help{
      "alpha parameter in exponential radial filter"};
  using group = Filtering;
};

struct RadialFilterHalfPower {
  using type = size_t;
  static constexpr OptionString help{
      "Half-power of the exponential radial filter argument"};
  using group = Filtering;
};

struct ObservationLMax {
  using type = size_t;
  static constexpr OptionString help{"Maximum l value for swsh output"};
  using group = Cce;
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr OptionString help{
      "Number of radial grid points in the spherical domain"};
  using group = Cce;
};

struct EndTime {
  using type = double;
  static constexpr OptionString help{"End time for the Cce Evolution."};
  static double default_value() noexcept {
    return std::numeric_limits<double>::infinity();
  }
  using group = Cce;
};

struct StartTime {
  using type = double;
  static constexpr OptionString help{
      "Cce Start time (default to earliest possible time)."};
  static double default_value() noexcept {
    return -std::numeric_limits<double>::infinity();
  }
  using group = Cce;
};

struct TargetStepSize {
  using type = double;
  static constexpr OptionString help{"Target time step size for Cce Evolution"};
  using group = Cce;
};

struct BoundaryDataFilename {
  using type = std::string;
  static constexpr OptionString help{
      "H5 file to read the wordltube data from."};
  using group = Cce;
};

struct H5LookaheadTimes {
  using type = size_t;
  static constexpr OptionString help{
      "Number of times steps from the h5 to cache each read."};
  static size_t default_value() noexcept { return 200; }
  using group = Cce;
};

struct H5Interpolator {
  using type = std::unique_ptr<Interpolator>;
  static constexpr OptionString help{
      "The interpolator for imported h5 worldtube data."};
  using group = Cce;
};

struct ScriInterpolationOrder {
  static std::string name() noexcept { return "ScriInterpOrder"; }
  using type = size_t;
  static constexpr OptionString help{"Order of time interpolation at scri+."};
  static size_t default_value() noexcept { return 5; }
  using group = Cce;
};

struct ScriOutputDensity {
  using type = size_t;
  static constexpr OptionString help{
      "Number of scri output points per timestep."};
  static size_t default_value() noexcept { return 1; }
  using group = Cce;
};

}  // namespace OptionTags

namespace InitializationTags {
/// An initialization tag that constructs a `WorldtubeDataManager` from options
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = WorldtubeDataManager;
  using option_tags =
      tmpl::list<OptionTags::LMax, OptionTags::BoundaryDataFilename,
                 OptionTags::H5LookaheadTimes, OptionTags::H5Interpolator>;

  static WorldtubeDataManager create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator) noexcept {
    return WorldtubeDataManager{
        std::make_unique<SpecWorldtubeH5BufferUpdater>(filename), l_max,
        number_of_lookahead_times, interpolator->get_clone()};
  }
};

struct LMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::LMax>;

  static size_t create_from_options(const size_t l_max) noexcept {
    return l_max;
  }
};

struct NumberOfRadialPoints : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::NumberOfRadialPoints>;

  static size_t create_from_options(
      const size_t number_of_radial_points) noexcept {
    return number_of_radial_points;
  }
};

struct StartTime : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::StartTime, OptionTags::BoundaryDataFilename>;

  static double create_from_options(double start_time,
                                    const std::string& filename) noexcept {
    if (start_time == -std::numeric_limits<double>::infinity()) {
      SpecWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      start_time = time_buffer[0];
    }
    return start_time;
  }
};

struct ScriInterpolationOrder : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriInterpolationOrder>;

  static size_t create_from_options(
      const size_t scri_plus_interpolation_order) noexcept {
    return scri_plus_interpolation_order;
  }
};

struct TargetStepSize : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::TargetStepSize>;

  static double create_from_options(const double target_step_size) noexcept {
    return target_step_size;
  }
};

struct EndTime : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::EndTime, OptionTags::BoundaryDataFilename>;

  static double create_from_options(double end_time,
                                    const std::string& filename) {
    if (end_time == std::numeric_limits<double>::infinity()) {
      SpecWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      end_time = time_buffer[time_buffer.size() - 1];
    }
    return end_time;
  }
};

struct ScriOutputDensity : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriOutputDensity>;

  static size_t create_from_options(const size_t scri_output_density) noexcept {
    return scri_output_density;
  }
};
}  // namespace InitializationTags

namespace Tags {
struct FilterLMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::FilterLMax>;

  static size_t create_from_options(const size_t filter_l_max) noexcept {
    return filter_l_max;
  }
};

struct RadialFilterAlpha : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RadialFilterAlpha>;

  static double create_from_options(const double radial_filter_alpha) noexcept {
    return radial_filter_alpha;
  }
};

struct RadialFilterHalfPower : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::RadialFilterHalfPower>;

  static size_t create_from_options(
      const size_t radial_filter_half_power) noexcept {
    return radial_filter_half_power;
  }
};
}  // namespace Tags
}  // namespace Cce
