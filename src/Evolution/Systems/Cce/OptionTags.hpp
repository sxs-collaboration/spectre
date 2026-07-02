// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "Evolution/Systems/Cce/AnalyticBoundaryDataManager.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/ExtractionRadius.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/InitializationTag.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
/// \brief %Option tags for CCE
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
      "SpEC worldtube data)."};
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

struct KleinGordonBoundaryDataFilename {
  using type = std::string;
  static constexpr Options::String help{
      "H5 file to read the Klein-Gordon wordltube data from. It could be the "
      "same as/different from `BoundaryDataFilename`."};
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

template <bool evolve_ccm>
struct InitializeJ {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ<evolve_ccm>>;
  static constexpr Options::String help{
      "The initialization for the first hypersurface for J"};
  using group = Cce;
};

/// Option for choosing the first-hypersurface initialization of J in an
/// analytic-solution CCE run.
///
/// \details Set to `FromAnalyticSolution` to use the J initialization provided
/// by the analytic solution itself (via
/// `Cce::Solutions::WorldtubeData::get_initialize_j`). Otherwise, any
/// option-creatable `Cce::InitializeJ::InitializeJ<false>` may be specified to
/// override the solution-provided initialization, exactly as in a standard CCE
/// run.
struct AnalyticInitializeJ {
  static std::string name() { return "InitializeJ"; }
  struct FromAnalyticSolution {};
  using type =
      Options::Auto<std::unique_ptr<::Cce::InitializeJ::InitializeJ<false>>,
                    FromAnalyticSolution>;
  static constexpr Options::String help{
      "The initialization for the first hypersurface for J. Set to "
      "'FromAnalyticSolution' to use the initialization provided by the "
      "analytic solution."};
  using group = Cce;
};
}  // namespace OptionTags

/// \brief Initialization tags for CCE
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
struct ExtractionRadius : db::SimpleTag {
  using type = double;
};

struct ExtractionRadiusSimple : ExtractionRadius {
  using base = ExtractionRadius;
  using option_tags = tmpl::list<OptionTags::ExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double extraction_radius) {
    return extraction_radius;
  }
};

struct ExtractionRadiusFromH5 : ExtractionRadius {
  using base = ExtractionRadius;
  using option_tags = tmpl::list<OptionTags::BoundaryDataFilename,
                                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(
      const std::string& filename,
      const std::optional<double>& extraction_radius) {
    const std::optional<double> radius =
        Cce::get_extraction_radius(filename, extraction_radius);
    return radius.value();
  }
};

struct FilePrefix : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::BondiSachsOutputFilePrefix>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

/// Tag for duplicating functionality of another tag, but allows creation from
/// options in the Cce::Evolution option group.
/// @{
template <typename Tag>
struct CceEvolutionPrefix;

namespace CceEvolutionPrefix_detail {
template <typename Tag>
struct CopyBase : db::SimpleTag {};

template <typename Tag>
  requires requires { typename Tag::base; }
struct CopyBase<Tag> : CceEvolutionPrefix<typename Tag::base> {
  using base = CceEvolutionPrefix<typename Tag::base>;
};
}  // namespace CceEvolutionPrefix_detail

template <db::simple_tag Tag>
struct CceEvolutionPrefix<Tag> : CceEvolutionPrefix_detail::CopyBase<Tag> {
  using type = typename Tag::type;
  static std::string name() { return db::tag_name<Tag>(); }
};

template <Parallel::untemplated_initialization_tag Tag>
struct CceEvolutionPrefix<Tag> : CceEvolutionPrefix_detail::CopyBase<Tag> {
  using type = typename Tag::type;
  using option_tags = db::wrap_tags_in<OptionTags::CceEvolutionPrefix,
                                       typename Tag::option_tags>;
  static std::string name() { return db::tag_name<Tag>(); }

  static constexpr bool pass_metavariables = Tag::pass_metavariables;
  template <typename... Args>
  static type create_from_options(const Args&... args) {
    return Tag::create_from_options(args...);
  }
};

template <Parallel::templated_initialization_tag Tag>
struct CceEvolutionPrefix<Tag> : CceEvolutionPrefix_detail::CopyBase<Tag> {
  using type = typename Tag::type;
  template <typename Metavariables>
  using option_tags =
      db::wrap_tags_in<OptionTags::CceEvolutionPrefix,
                       typename Tag::template option_tags<Metavariables>>;
  static std::string name() { return db::tag_name<Tag>(); }

  static constexpr bool pass_metavariables = Tag::pass_metavariables;
  template <typename Metavariables, typename... Args>
  static type create_from_options(const Args&... args) {
    return Tag::template create_from_options<Metavariables>(args...);
  }
};

template <db::reference_tag Tag>
struct CceEvolutionPrefix<Tag> : CceEvolutionPrefix<typename Tag::base>,
                                 db::ReferenceTag {
  using base = CceEvolutionPrefix<typename Tag::base>;
  using argument_tags =
      tmpl::transform<typename Tag::argument_tags,
                      tmpl::bind<CceEvolutionPrefix, tmpl::_1>>;
  template <typename... Args>
  static const auto& get(const Args&... args) {
    return Tag::get(args...);
  }
};
/// @}

/// A tag that constructs a `MetricWorldtubeDataManager` or
/// `BondiWorldtubeDataManager` from options
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = std::unique_ptr<WorldtubeDataManager<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>>;
  using option_tags =
      tmpl::list<Spectral::Swsh::OptionTags::LMax,
                 OptionTags::BoundaryDataFilename, OptionTags::H5LookaheadTimes,
                 OptionTags::H5Interpolator,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
      const std::optional<double> extraction_radius) {
    const std::string text_radius_str = Cce::get_text_radius(filename);
    try {
      // If this doesn't throw an exception, then an extraction radius was
      // supplied in the filename. We don't actually need the value.
      const double text_radius = std::stod(text_radius_str);
      (void)text_radius;
      if (extraction_radius.has_value()) {
        Parallel::printf(
            "Warning: Option ExtractionRadius is set to a specific value and "
            "there is an extraction radius in the H5 filename. The value in "
            "the file name will be ignored.It is recommended to set "
            "`ExtractionRadius` to `\"Auto\"` if the H5 filename has the "
            "extraction radius in it to make the input file clearer.\n");
      }
    } catch (const std::invalid_argument&) {
    }

    return std::make_unique<BondiWorldtubeDataManager>(
        std::make_unique<BondiWorldtubeH5BufferUpdater<ComplexModalVector>>(
            filename, extraction_radius),
        l_max, number_of_lookahead_times, interpolator->get_clone());
  }
};

/// A tag that constructs a `KleinGordonWorldtubeDataManager` from options
struct KleinGordonH5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = std::unique_ptr<
      WorldtubeDataManager<Tags::klein_gordon_worldtube_boundary_tags>>;
  using option_tags =
      tmpl::list<Spectral::Swsh::OptionTags::LMax,
                 OptionTags::KleinGordonBoundaryDataFilename,
                 OptionTags::H5LookaheadTimes, OptionTags::H5Interpolator,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
      const std::optional<double> extraction_radius) {
    return std::make_unique<KleinGordonWorldtubeDataManager>(
        std::make_unique<KleinGordonWorldtubeH5BufferUpdater>(
            filename, extraction_radius),
        l_max, number_of_lookahead_times, interpolator->get_clone());
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
struct StartTimeFromFile : Tags::StartTime {
  using base = Tags::StartTime;
  using option_tags =
      tmpl::list<OptionTags::StartTime, OptionTags::BoundaryDataFilename,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(
      const std::optional<double> start_time, const std::string& filename,
      const std::optional<double>& extraction_radius) {
    if (start_time.has_value()) {
      return *start_time;
    }

    BondiWorldtubeH5BufferUpdater<ComplexModalVector> h5_boundary_updater{
        filename, extraction_radius};
    const auto& time_buffer = h5_boundary_updater.get_time_buffer();
    return time_buffer[0];
  }
};

/// \brief Represents the start time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedStartTime : Tags::StartTime {
  using base = Tags::StartTime;
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
struct EndTimeFromFile : Tags::EndTime {
  using base = Tags::EndTime;
  using option_tags =
      tmpl::list<OptionTags::EndTime, OptionTags::BoundaryDataFilename,
                 OptionTags::StandaloneExtractionRadius>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(
      const std::optional<double> end_time, const std::string& filename,
      const std::optional<double>& extraction_radius) {
    if (end_time) {
      return *end_time;
    }
    BondiWorldtubeH5BufferUpdater<ComplexModalVector> h5_boundary_updater{
        filename, extraction_radius};
    const auto& time_buffer = h5_boundary_updater.get_time_buffer();
    return time_buffer[time_buffer.size() - 1];
  }
};

/// \brief Represents the final time of a CCE evolution that should just proceed
/// until it receives no more boundary data and becomes quiescent.
struct NoEndTime : Tags::EndTime {
  using base = Tags::EndTime;
  using option_tags = tmpl::list<>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options() {
    return std::numeric_limits<double>::infinity();
  }
};

/// \brief Represents the final time of a bounded CCE evolution that must be
/// supplied in the input file (for e.g. analytic tests).
struct SpecifiedEndTime : Tags::EndTime {
  using base = Tags::EndTime;
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

/// Tag for first-hypersurface initialization procedure specified by input
/// options.
template <bool evolve_ccm>
struct InitializeJ : db::SimpleTag {
  using type = std::unique_ptr<::Cce::InitializeJ::InitializeJ<evolve_ccm>>;
  using option_tags = tmpl::list<OptionTags::InitializeJ<evolve_ccm>>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::Cce::InitializeJ::InitializeJ<evolve_ccm>>
  create_from_options(
      const std::unique_ptr<::Cce::InitializeJ::InitializeJ<evolve_ccm>>&
          initialize_j) {
    return initialize_j->get_clone();
  }
};

// Tag that generates an `Cce::InitializeJ::InitializeJ` derived class for an
// analytic-solution run. By default the initialization is provided by the
// analytic solution itself, but it may be overridden from the input file via
// `OptionTags::AnalyticInitializeJ`.
struct AnalyticInitializeJ : InitializeJ<false> {
  using base = InitializeJ<false>;
  using option_tags =
      tmpl::list<OptionTags::AnalyticInitializeJ, OptionTags::AnalyticSolution,
                 OptionTags::StartTime>;
  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<::Cce::InitializeJ::InitializeJ<false>>
  create_from_options(
      const std::optional<std::unique_ptr<
          ::Cce::InitializeJ::InitializeJ<false>>>& initialize_j,
      const std::unique_ptr<Cce::Solutions::WorldtubeData>& worldtube_data,
      const std::optional<double> start_time) {
    if (initialize_j.has_value()) {
      return initialize_j.value()->get_clone();
    }
    return worldtube_data->get_initialize_j(*start_time);
  }
};

/// A tag that constructs a `AnalyticBoundaryDataManager` from options
struct AnalyticBoundaryDataManager : db::SimpleTag {
  using type = ::Cce::AnalyticBoundaryDataManager;
  using option_tags =
      tmpl::list<OptionTags::ExtractionRadius, Spectral::Swsh::OptionTags::LMax,
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
