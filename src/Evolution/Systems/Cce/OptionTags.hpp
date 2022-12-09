// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/Printf.hpp"
#include "Time/Tags.hpp"
#include "Utilities/PrettyType.hpp"

namespace Cce {
namespace OptionTags {

/// %Option group
struct Cce {
  static constexpr Options::String help = {
      "Options for the Cce evolution system"};
};

/// %Option group
struct Filtering {
  static constexpr Options::String help = {"Options for the filtering in Cce"};
  using group = Cce;
};

/// %Option group for evolution-related quantities in the CCE system
struct Evolution {
  static constexpr Options::String help = {"Options for the CCE evolution"};
  using group = Cce;
};

/// A prefix for common tags (e.g. from Time/Tags.hpp) that are specific to CCE,
/// so should be in the Cce::Evolution group.
template <typename OptionTag>
struct CceEvolutionPrefix {
  using type = typename OptionTag::type;
  static std::string name() { return pretty_type::name<OptionTag>(); }
  static constexpr Options::String help = OptionTag::help;
  using group = Evolution;
};

struct BondiSachsOutputFilePrefix {
  using type = std::string;
  static constexpr Options::String help{
      "Filename prefix for dumping Bondi-Sachs data on worltube radii. Files "
      "will have this prefix prepended to 'CceRXXXX.h5' where XXXX will be the "
      "zero-padded extraction radius to the nearest integer."};
  using group = Cce;
};

struct LMax {
  using type = size_t;
  static constexpr Options::String help{
      "Maximum l value for spin-weighted spherical harmonics"};
  using group = Cce;
};

struct FilterLMax {
  using type = size_t;
  static constexpr Options::String help{"l mode cutoff for angular filtering"};
  using group = Filtering;
};

struct RadialFilterAlpha {
  using type = double;
  static constexpr Options::String help{
      "alpha parameter in exponential radial filter"};
  using group = Filtering;
};

struct RadialFilterHalfPower {
  using type = size_t;
  static constexpr Options::String help{
      "Half-power of the exponential radial filter argument"};
  using group = Filtering;
};

struct ObservationLMax {
  using type = size_t;
  static constexpr Options::String help{"Maximum l value for swsh output"};
  using group = Cce;
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr Options::String help{
      "Number of radial grid points in the spherical domain"};
  using group = Cce;
};

struct ExtractionRadius {
  using type = double;
  static constexpr Options::String help{"Extraction radius of the CCE system."};
  using group = Cce;
};

struct StandaloneExtractionRadius {
  static std::string name() { return "ExtractionRadius"; }
  using type = Options::Auto<double>;

  static constexpr Options::String help{
      "Extraction radius of the CCE system for a standalone run. This may be "
      "set to \"Auto\" to infer the radius from the filename (often used for "
      "SpEC worldtube data). This option is unused if `H5IsBondiData` is "
      "`true`, and should be \"Auto\" for such runs."};
  using group = Cce;
};

struct EndTime {
  using type = Options::Auto<double>;
  static constexpr Options::String help{"End time for the Cce Evolution."};
  static type suggested_value() { return {}; }
  using group = Cce;
};

struct StartTime {
  using type = Options::Auto<double>;
  static constexpr Options::String help{
      "Cce Start time (default to earliest possible time)."};
  static type suggested_value() { return {}; }
  using group = Cce;
};

struct BoundaryDataFilename {
  using type = std::string;
  static constexpr Options::String help{
      "H5 file to read the wordltube data from."};
  using group = Cce;
};

struct H5LookaheadTimes {
  using type = size_t;
  static constexpr Options::String help{
      "Number of times steps from the h5 to cache each read."};
  static size_t suggested_value() { return 200; }
  using group = Cce;
};

struct H5Interpolator {
  using type = std::unique_ptr<intrp::SpanInterpolator>;
  static constexpr Options::String help{
      "The interpolator for imported h5 worldtube data."};
  using group = Cce;
};

struct H5IsBondiData {
  using type = bool;
  static constexpr Options::String help{
      "true for boundary data in Bondi form, false for metric data. Metric "
      "data is more readily available from Cauchy simulations, so historically "
      "has been the typical format provided by SpEC simulations. Bondi data is "
      "much more efficient for storage size and performance, but both must be "
      "supported for compatibility with current CCE data sources."};
  using group = Cce;
};

struct FixSpecNormalization {
  using type = bool;
  static constexpr Options::String help{
      "Set to true if corrections for SpEC data impurities should be applied "
      "automatically based on the `VersionHist.ver` data set in the H5. "
      "Typically, this should be set to true if the metric data is created "
      "from SpEC, and false otherwise."};
  using group = Cce;
};

struct AnalyticSolution {
  using type = std::unique_ptr<Solutions::WorldtubeData>;
  static constexpr Options::String help{
      "Analytic worldtube data for tests of CCE."};
  using group = Cce;
};

struct GhInterfaceManager {
  using type = InterfaceManagers::GhLocalTimeStepping;
  static constexpr Options::String help{
      "Class to manage worldtube data from a GH system."};
  using group = Cce;
};

struct ScriInterpolationOrder {
  static std::string name() { return "ScriInterpOrder"; }
  using type = size_t;
  static constexpr Options::String help{
      "Order of time interpolation at scri+."};
  static size_t suggested_value() { return 5; }
  using group = Cce;
};

struct ScriOutputDensity {
  using type = size_t;
  static constexpr Options::String help{
      "Number of scri output points per timestep."};
  static size_t suggested_value() { return 1; }
  using group = Cce;
};

template <bool uses_partially_flat_cartesian_coordinates>
struct InitializeJ {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ<
      uses_partially_flat_cartesian_coordinates>>;
  static constexpr Options::String help{
      "The initialization for the first hypersurface for J"};
  using group = Cce;
};
}  // namespace OptionTags

namespace InitializationTags {
struct ScriInterpolationOrder : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriInterpolationOrder>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(
      const size_t scri_plus_interpolation_order) {
    return scri_plus_interpolation_order;
  }
};

struct ExtractionRadius : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double extraction_radius) {
    return extraction_radius;
  }
};

struct ScriOutputDensity : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ScriOutputDensity>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t scri_output_density) {
    return scri_output_density;
  }
};
}  // namespace InitializationTags

namespace Tags {
struct FilePrefix : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::BondiSachsOutputFilePrefix>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

/// Tag for duplicating functionality of another tag, but allows creation from
/// options in the Cce::Evolution option group.
template <typename Tag>
struct CceEvolutionPrefix : Tag {
  using type = typename Tag::type;
  using option_tags = db::wrap_tags_in<OptionTags::CceEvolutionPrefix,
                                       typename Tag::option_tags>;
  static std::string name() { return pretty_type::name<Tag>(); }

  static constexpr bool pass_metavariables = Tag::pass_metavariables;
  template <typename Metavariables, typename... Args>
  static type create_from_options(const Args&... args) {
    return Tag::template create_from_options<Metavariables>(args...);
  }

  template <typename... Args>
  static type create_from_options(const Args&... args) {
    return Tag::create_from_options(args...);
  }
};

/// A tag that constructs a `MetricWorldtubeDataManager` from options
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = std::unique_ptr<WorldtubeDataManager>;
  using option_tags =
      tmpl::list<OptionTags::LMax, OptionTags::BoundaryDataFilename,
                 OptionTags::H5LookaheadTimes, OptionTags::H5Interpolator,
                 OptionTags::H5IsBondiData, OptionTags::FixSpecNormalization,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
      const bool h5_is_bondi_data, const bool fix_spec_normalization,
      const std::optional<double> extraction_radius) {
    if (h5_is_bondi_data) {
      if (static_cast<bool>(extraction_radius)) {
        Parallel::printf(
            "Warning: Option ExtractionRadius is set to a specific value and "
            "H5IsBondiData is set to `true` -- the ExtractionRadius will not "
            "be used, because all radius information is specified in the input "
            "file for the Bondi worldtube data format. It is recommended to "
            "set `ExtractionRadius` to `\"Auto\"` to make the input file "
            "clearer.\n");
      }
      return std::make_unique<BondiWorldtubeDataManager>(
          std::make_unique<BondiWorldtubeH5BufferUpdater>(filename,
                                                          extraction_radius),
          l_max, number_of_lookahead_times, interpolator->get_clone());
    } else {
      return std::make_unique<MetricWorldtubeDataManager>(
          std::make_unique<MetricWorldtubeH5BufferUpdater>(filename,
                                                           extraction_radius),
          l_max, number_of_lookahead_times, interpolator->get_clone(),
          fix_spec_normalization);
    }
  }
};

struct LMax : db::SimpleTag, Spectral::Swsh::Tags::LMaxBase {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::LMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t l_max) { return l_max; }
};

struct NumberOfRadialPoints : db::SimpleTag,
                              Spectral::Swsh::Tags::NumberOfRadialPointsBase {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::NumberOfRadialPoints>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t number_of_radial_points) {
    return number_of_radial_points;
  }
};

struct ObservationLMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::ObservationLMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t observation_l_max) {
    return observation_l_max;
  }
};

struct FilterLMax : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::FilterLMax>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t filter_l_max) {
    return filter_l_max;
  }
};

struct RadialFilterAlpha : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RadialFilterAlpha>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double radial_filter_alpha) {
    return radial_filter_alpha;
  }
};

struct RadialFilterHalfPower : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::RadialFilterHalfPower>;

  static constexpr bool pass_metavariables = false;
  static size_t create_from_options(const size_t radial_filter_half_power) {
    return radial_filter_half_power;
  }
};

/// \brief Represents the start time of a bounded CCE evolution, determined
/// either from option specification or from the file
///
/// \details If no start time is specified in the input file (so the option
/// `OptionTags::StartTime` is set to "Auto"), this will find the start time
/// from the provided H5 file. If `OptionTags::StartTime` takes any other value,
/// it will be used directly as the start time for the CCE evolution instead.
struct StartTimeFromFile : Tags::StartTime, db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::StartTime, OptionTags::BoundaryDataFilename,
                 OptionTags::H5IsBondiData>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> start_time,
                                    const std::string& filename,
                                    const bool is_bondi_data) {
    if (start_time) {
      return *start_time;
    }
    if (is_bondi_data) {
      BondiWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[0];
    } else {
      MetricWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[0];
    }
  }
};

/// \brief Represents the start time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedStartTime : Tags::StartTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::StartTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> start_time) {
    if (not start_time.has_value()) {
      ERROR(
          "The start time must be explicitly specified for the tag "
          "`SpecifiedStartTime`");
    }
    return *start_time;
  }
};

/// \brief Represents the final time of a bounded CCE evolution, determined
/// either from option specification or from the file
///
/// \details If no end time is specified in the input file (so the option
/// `OptionTags::EndTime` is set to "Auto"), this will find the end time
/// from the provided H5 file. If `OptionTags::EndTime` takes any other value,
/// it will be used directly as the final time for the CCE evolution instead.
struct EndTimeFromFile : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::EndTime, OptionTags::BoundaryDataFilename,
                 OptionTags::H5IsBondiData>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> end_time,
                                    const std::string& filename,
                                    const bool is_bondi_data) {
    if (end_time) {
      return *end_time;
    }
    if (is_bondi_data) {
      BondiWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[time_buffer.size() - 1];
    } else {
      MetricWorldtubeH5BufferUpdater h5_boundary_updater{filename};
      const auto& time_buffer = h5_boundary_updater.get_time_buffer();
      return time_buffer[time_buffer.size() - 1];
    }
  }
};

/// \brief Represents the final time of a CCE evolution that should just proceed
/// until it receives no more boundary data and becomes quiescent.
struct NoEndTime : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options() {
    return std::numeric_limits<double>::infinity();
  }
};

/// \brief Represents the final time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedEndTime : Tags::EndTime, db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::EndTime>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const std::optional<double> end_time) {
    if (not end_time.has_value()) {
      ERROR(
          "The end time must be explicitly specified for the tag "
          "`SpecifiedEndTime`");
    }
    return *end_time;
  }
};

struct GhInterfaceManager : db::SimpleTag {
  using type = InterfaceManagers::GhLocalTimeStepping;
  using option_tags = tmpl::list<OptionTags::GhInterfaceManager>;

  static constexpr bool pass_metavariables = false;
  static InterfaceManagers::GhLocalTimeStepping create_from_options(
      const InterfaceManagers::GhLocalTimeStepping& interface_manager) {
    return interface_manager;
  }
};

/// Base tag for first-hypersurface initialization procedure
struct InitializeJBase : db::BaseTag {};

/// Tag for first-hypersurface initialization procedure specified by input
/// options.
template <bool uses_partially_flat_cartesian_coordinates>
struct InitializeJ : db::SimpleTag, InitializeJBase {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ<
      uses_partially_flat_cartesian_coordinates>>;
  using option_tags = tmpl::list<
      OptionTags::InitializeJ<uses_partially_flat_cartesian_coordinates>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::Cce::InitializeJ::InitializeJ<
      uses_partially_flat_cartesian_coordinates>>
  create_from_options(
      const std::unique_ptr<::Cce::InitializeJ::InitializeJ<
          uses_partially_flat_cartesian_coordinates>>& initialize_j) {
    return initialize_j->get_clone();
  }
};

// Tags that generates an `Cce::InitializeJ::InitializeJ` derived class from an
// analytic solution.
struct AnalyticInitializeJ : db::SimpleTag, InitializeJBase {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ<false>>;
  using option_tags =
      tmpl::list<OptionTags::AnalyticSolution, OptionTags::StartTime>;
  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::Cce::InitializeJ::InitializeJ<false>>
  create_from_options(
      const std::unique_ptr<Cce::Solutions::WorldtubeData>& worldtube_data,
      const std::optional<double> start_time) {
    return worldtube_data->get_initialize_j(*start_time);
  }
};

/// A tag that constructs a `AnalyticBoundaryDataManager` from options
struct AnalyticBoundaryDataManager : db::SimpleTag {
  using type = ::Cce::AnalyticBoundaryDataManager;
  using option_tags = tmpl::list<OptionTags::ExtractionRadius, OptionTags::LMax,
                                 OptionTags::AnalyticSolution>;

  static constexpr bool pass_metavariables = false;
  static Cce::AnalyticBoundaryDataManager create_from_options(
      const double extraction_radius, const size_t l_max,
      const std::unique_ptr<Cce::Solutions::WorldtubeData>& worldtube_data) {
    return ::Cce::AnalyticBoundaryDataManager(l_max, extraction_radius,
                                              worldtube_data->get_clone());
  }
};

/// Represents whether the news should be provided at noninertial times.
///
/// \details Currently, this is only useful for analytic solutions for which the
/// inertial-time news is difficult to compute.
struct OutputNoninertialNews : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::AnalyticSolution>;
  static constexpr bool pass_metavariables = false;
  static bool create_from_options(
      const std::unique_ptr<Cce::Solutions::WorldtubeData>& worldtube_data) {
    return worldtube_data->use_noninertial_news();
  }
};
}  // namespace Tags
}  // namespace Cce
