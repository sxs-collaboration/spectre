// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ExtractionRadius.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "IO/H5/CombineH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
// Convenient tag lists
using modal_metric_input_tags = Cce::cce_metric_input_tags<ComplexModalVector>;
using nodal_metric_input_tags = Cce::cce_metric_input_tags<DataVector>;
using modal_bondi_input_tags = Cce::Tags::worldtube_boundary_tags_for_writing<
    Spectral::Swsh::Tags::SwshTransform>;
using nodal_bondi_input_tags =
    Cce::Tags::worldtube_boundary_tags_for_writing<Cce::Tags::BoundaryValue>;

using AdmOptions = Cce::MetricWorldtubeH5BufferUpdater<DataVector>::AdmOptions;

// from a data-varies-fastest set of buffers provided by
// `MetricWorldtubeH5BufferUpdater` extract the set of coefficients for a
// particular time given by `buffer_time_offset` into the `time_span` size of
// buffer.
void slice_buffers_to_libsharp_modes(
    const gsl::not_null<Variables<modal_metric_input_tags>*> coefficients_set,
    const Variables<modal_metric_input_tags>& coefficients_buffers,
    const size_t buffer_time_offset, const size_t computation_l_max) {
  SpinWeighted<ComplexModalVector, 0> goldberg_mode_buffer;
  SpinWeighted<ComplexModalVector, 0> libsharp_mode_buffer;

  const size_t goldberg_mode_size = square(computation_l_max + 1);
  const size_t libsharp_mode_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(computation_l_max);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      tmpl::for_each<Cce::Tags::detail::apply_derivs_t<
          Cce::Tags::detail::SpatialMetric<ComplexModalVector>>>(
          [&](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            const auto& all_goldberg_modes =
                get<tag>(coefficients_buffers).get(i, j);
            auto& all_libsharp_modes = get<tag>(*coefficients_set).get(i, j);

            // NOLINTBEGIN
            goldberg_mode_buffer.set_data_ref(
                const_cast<ComplexModalVector&>(all_goldberg_modes).data() +
                    buffer_time_offset * goldberg_mode_size,
                goldberg_mode_size);
            libsharp_mode_buffer.set_data_ref(all_libsharp_modes.data(),
                                              libsharp_mode_size);
            // NOLINTEND

            Spectral::Swsh::goldberg_to_libsharp_modes(
                make_not_null(&libsharp_mode_buffer), goldberg_mode_buffer,
                computation_l_max);
          });
    }
    tmpl::for_each<Cce::Tags::detail::apply_derivs_t<
        Cce::Tags::detail::Shift<ComplexModalVector>>>([&](auto tag_v) {
      using tag = typename decltype(tag_v)::type;
      const auto& all_goldberg_modes = get<tag>(coefficients_buffers).get(i);
      auto& all_libsharp_modes = get<tag>(*coefficients_set).get(i);

      // NOLINTBEGIN
      goldberg_mode_buffer.set_data_ref(
          const_cast<ComplexModalVector&>(all_goldberg_modes).data() +
              buffer_time_offset * goldberg_mode_size,
          goldberg_mode_size);
      libsharp_mode_buffer.set_data_ref(all_libsharp_modes.data(),
                                        libsharp_mode_size);
      // NOLINTEND

      Spectral::Swsh::goldberg_to_libsharp_modes(
          make_not_null(&libsharp_mode_buffer), goldberg_mode_buffer,
          computation_l_max);
    });
  }
  tmpl::for_each<Cce::Tags::detail::apply_derivs_t<
      Cce::Tags::detail::Lapse<ComplexModalVector>>>([&](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    const auto& all_goldberg_modes = get(get<tag>(coefficients_buffers));
    auto& all_libsharp_modes = get(get<tag>(*coefficients_set));

    // NOLINTBEGIN
    goldberg_mode_buffer.set_data_ref(
        const_cast<ComplexModalVector&>(all_goldberg_modes).data() +
            buffer_time_offset * goldberg_mode_size,
        goldberg_mode_size);
    libsharp_mode_buffer.set_data_ref(all_libsharp_modes.data(),
                                      libsharp_mode_size);
    // NOLINTEND

    Spectral::Swsh::goldberg_to_libsharp_modes(
        make_not_null(&libsharp_mode_buffer), goldberg_mode_buffer,
        computation_l_max);
  });
}

template <typename BoundaryData>
void write_bondi_data_to_disk(
    const gsl::not_null<Cce::WorldtubeModeRecorder*> recorder,
    const BoundaryData& nodal_boundary_data, const double time,
    const size_t data_l_max) {
  tmpl::for_each<nodal_bondi_input_tags>([&](auto tag_v) {
    using tag = typename decltype(tag_v)::type;

    const ComplexDataVector& nodal_data =
        get(get<tag>(nodal_boundary_data)).data();

    recorder->append_modal_data<tag::tag::type::type::spin>(
        Cce::dataset_label_for_tag<typename tag::tag>(), time, nodal_data,
        data_l_max);
  });
}

// read in the data from a (previously standard) SpEC worldtube file
// `input_file`, perform the boundary computation, and dump the (considerably
// smaller) dataset associated with the spin-weighted scalars to
// `output_file`.
void perform_cce_worldtube_reduction(
    const std::string& input_file, const std::string& output_file,
    const size_t input_buffer_depth, const size_t l_max_factor,
    const std::optional<double>& extraction_radius,
    const bool fix_spec_normalization, const bool descending_m) {
  Cce::MetricWorldtubeH5BufferUpdater<ComplexModalVector> buffer_updater{
      input_file, extraction_radius, descending_m};
  const size_t l_max = buffer_updater.get_l_max();
  // Perform the boundary computation to scalars at some factor > 1 of the input
  // l_max to be absolutely certain that there are no problems associated with
  // aliasing.
  const size_t computation_l_max = l_max_factor * l_max;

  const DataVector& time_buffer = buffer_updater.get_time_buffer();
  // We're not interpolating in time, this is just a reasonable number of rows
  // to ingest at a time. If the buffer depth from the input file is larger than
  // the number of times we have, just use the number of times
  const size_t buffer_depth = std::min(time_buffer.size(), input_buffer_depth);
  const size_t size_of_buffer = square(computation_l_max + 1) * buffer_depth;

  Variables<modal_metric_input_tags> coefficients_buffers{size_of_buffer};
  Variables<modal_metric_input_tags> coefficients_set{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(computation_l_max)};

  Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>>
      boundary_data_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(computation_l_max)};

  size_t time_span_start = 0;
  size_t time_span_end = 0;
  Cce::WorldtubeModeRecorder recorder{l_max, output_file};

  for (size_t i = 0; i < time_buffer.size(); ++i) {
    const double time = time_buffer[i];
    buffer_updater.update_buffers_for_time(
        make_not_null(&coefficients_buffers), make_not_null(&time_span_start),
        make_not_null(&time_span_end), time, computation_l_max, 0, buffer_depth,
        false);

    slice_buffers_to_libsharp_modes(make_not_null(&coefficients_set),
                                    coefficients_buffers, i - time_span_start,
                                    computation_l_max);

    const auto create_boundary_data = [&](const auto&... tags) {
      if (not buffer_updater.has_version_history() and fix_spec_normalization) {
        Cce::create_bondi_boundary_data_from_unnormalized_spec_modes(
            make_not_null(&boundary_data_variables),
            get<tmpl::type_from<std::decay_t<decltype(tags)>>>(
                coefficients_set)...,
            buffer_updater.get_extraction_radius(), computation_l_max);
      } else {
        Cce::create_bondi_boundary_data(
            make_not_null(&boundary_data_variables),
            get<tmpl::type_from<std::decay_t<decltype(tags)>>>(
                coefficients_set)...,
            buffer_updater.get_extraction_radius(), computation_l_max);
      }
    };

    tmpl::as_pack<modal_metric_input_tags>(create_boundary_data);

    write_bondi_data_to_disk(make_not_null(&recorder), boundary_data_variables,
                             time, computation_l_max);
  }
}

template <typename BoundaryTags>
tuples::tagged_tuple_from_typelist<nodal_bondi_input_tags>
create_bondi_nodal_views(const Variables<BoundaryTags>& bondi_boundary_data,
                         const size_t time_offset, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tuples::tagged_tuple_from_typelist<nodal_bondi_input_tags> result;

  tmpl::for_each<nodal_bondi_input_tags>([&](auto tag_v) {
    using tag = typename decltype(tag_v)::type;

    make_const_view(
        make_not_null(&std::as_const(get(tuples::get<tag>(result)).data())),
        get(get<tag>(bondi_boundary_data)).data(),
        time_offset * number_of_angular_points, number_of_angular_points);
  });

  return result;
}

template <typename BoundaryTags>
tuples::tagged_tuple_from_typelist<nodal_metric_input_tags>
create_metric_nodal_views(const Variables<BoundaryTags>& bondi_boundary_data,
                          const size_t time_offset, const size_t l_max) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tuples::tagged_tuple_from_typelist<nodal_metric_input_tags> result;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      tmpl::for_each<Cce::Tags::detail::apply_derivs_t<
          Cce::Tags::detail::SpatialMetric<DataVector>>>([&](auto tag_v) {
        using tag = typename decltype(tag_v)::type;

        make_const_view(
            make_not_null(&std::as_const(tuples::get<tag>(result).get(i, j))),
            get<tag>(bondi_boundary_data).get(i, j),
            time_offset * number_of_angular_points, number_of_angular_points);
      });
    }
    tmpl::for_each<Cce::Tags::detail::apply_derivs_t<
        Cce::Tags::detail::Shift<DataVector>>>([&](auto tag_v) {
      using tag = typename decltype(tag_v)::type;

      make_const_view(
          make_not_null(&std::as_const(tuples::get<tag>(result).get(i))),
          get<tag>(bondi_boundary_data).get(i),
          time_offset * number_of_angular_points, number_of_angular_points);
    });
  }

  tmpl::for_each<
      Cce::Tags::detail::apply_derivs_t<Cce::Tags::detail::Lapse<DataVector>>>(
      [&](auto tag_v) {
        using tag = typename decltype(tag_v)::type;

        make_const_view(
            make_not_null(&std::as_const(get(tuples::get<tag>(result)))),
            get(get<tag>(bondi_boundary_data)),
            time_offset * number_of_angular_points, number_of_angular_points);
      });

  return result;
}

void bondi_nodal_to_bondi_modal(
    const std::string& input_file, const std::string& output_file,
    const size_t input_buffer_depth,
    const std::optional<double>& extraction_radius) {
  Cce::BondiWorldtubeH5BufferUpdater<ComplexDataVector> buffer_updater{
      input_file, extraction_radius};
  const size_t l_max = buffer_updater.get_l_max();

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const DataVector& time_buffer = buffer_updater.get_time_buffer();
  // We're not interpolating in time, this is just a reasonable number of rows
  // to ingest at a time. If the buffer depth from the input file is larger than
  // the number of times we have, just use the number of times
  const size_t buffer_depth = std::min(time_buffer.size(), input_buffer_depth);
  const size_t size_of_buffer = buffer_depth * number_of_angular_points;

  Variables<nodal_bondi_input_tags> nodal_buffer{size_of_buffer};

  size_t time_span_start = 0;
  size_t time_span_end = 0;
  Cce::WorldtubeModeRecorder recorder{l_max, output_file};

  for (size_t i = 0; i < time_buffer.size(); i++) {
    const double time = time_buffer[i];
    buffer_updater.update_buffers_for_time(
        make_not_null(&nodal_buffer), make_not_null(&time_span_start),
        make_not_null(&time_span_end), time, l_max, 0, buffer_depth, false);

    const auto nodal_data_at_time =
        create_bondi_nodal_views(nodal_buffer, i - time_span_start, l_max);

    write_bondi_data_to_disk(make_not_null(&recorder), nodal_data_at_time, time,
                             l_max);
  }
}

void metric_nodal_to_bondi_modal(const std::string& input_file,
                                 const std::string& output_file,
                                 const size_t input_buffer_depth,
                                 const std::optional<double>& extraction_radius,
                                 const std::optional<AdmOptions>& adm_options) {
  Cce::MetricWorldtubeH5BufferUpdater<DataVector> buffer_updater{
      input_file, extraction_radius, false, adm_options};
  const size_t l_max = buffer_updater.get_l_max();

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  const DataVector& time_buffer = buffer_updater.get_time_buffer();
  // We're not interpolating in time, this is just a reasonable number of rows
  // to ingest at a time. If the buffer depth from the input file is larger than
  // the number of times we have, just use the number of times
  const size_t buffer_depth = std::min(time_buffer.size(), input_buffer_depth);
  const size_t size_of_buffer = buffer_depth * number_of_angular_points;

  Variables<nodal_metric_input_tags> nodal_buffer{size_of_buffer};

  Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>>
      boundary_data_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)};

  size_t time_span_start = 0;
  size_t time_span_end = 0;
  Cce::WorldtubeModeRecorder recorder{l_max, output_file};

  for (size_t i = 0; i < time_buffer.size(); i++) {
    const double time = time_buffer[i];

    buffer_updater.update_buffers_for_time(
        make_not_null(&nodal_buffer), make_not_null(&time_span_start),
        make_not_null(&time_span_end), time, l_max, 0, buffer_depth, false);

    const auto metric_nodal_data_at_time =
        create_metric_nodal_views(nodal_buffer, i - time_span_start, l_max);

    tmpl::as_pack<nodal_metric_input_tags>([&](const auto&... tags) {
      Cce::create_bondi_boundary_data(
          make_not_null(&boundary_data_variables),
          get<tmpl::type_from<std::decay_t<decltype(tags)>>>(
              metric_nodal_data_at_time)...,
          buffer_updater.get_extraction_radius(), l_max);
    });

    write_bondi_data_to_disk(make_not_null(&recorder), boundary_data_variables,
                             time, l_max);
  }
}

// If we use the AdmOptions class directly in an option tag, the input file
// would look like:
//
// InputDataFormat:
//   AdvectiveLapse: True
//   AdvectiveShift: True
//   AuxiliaryShiftFactor: False
//
// However, this doesn't have the name `AdmMetricNodal`. This is consistent with
// how our options work, but is confusing (especially for users who are
// unfamiliar with spectre) since the actual input data format isn't shown. This
// class now changes the input file to look like:
//
// InputDataFormat:
//   AdmMetricNodal:
//     AdvectiveLapse: True
//     AdvectiveShift: True
//     AuxiliaryShiftFactor: False
//
// which is much clearer.
struct AdmMetricNodalOptions {
  struct AdmMetricNodal {
    using type = ::AdmOptions;
    static constexpr Options::String help = ::AdmOptions::help;
  };

  using options = tmpl::list<AdmMetricNodal>;
  static constexpr Options::String help = ::AdmOptions::help;

  AdmMetricNodalOptions() = default;
  // NOLINTNEXTLINE
  AdmMetricNodalOptions(::AdmOptions adm_metric_nodal_in)
      : adm_metric_nodal(adm_metric_nodal_in) {}

  ::AdmOptions adm_metric_nodal;
};

enum class InputDataFormat {
  AdmMetricNodal,
  MetricNodal,
  MetricModal,
  BondiNodal,
  BondiModal
};

std::ostream& operator<<(std::ostream& os,
                         const InputDataFormat input_data_format) {
  switch (input_data_format) {
    case InputDataFormat::MetricNodal:
      return os << "MetricNodal";
    case InputDataFormat::MetricModal:
      return os << "MetricModal";
    case InputDataFormat::BondiNodal:
      return os << "BondiNodal";
    case InputDataFormat::BondiModal:
      return os << "BondiModal";
    default:
      ERROR("Unknown InputDataFormat type");
  }
}

namespace OptionTags {
struct InputH5Files {
  static std::string name() { return "InputH5File"; }
  using type = std::variant<std::string, std::vector<std::string>>;
  static constexpr Options::String help =
      "Name of H5 worldtube file(s). A '.h5' extension will be added if "
      "needed. Can specify a single file or if multiple files are specified, "
      "this will combine the times in each file. If there are "
      "duplicate/overlapping times, the last/latest of the times are chosen.";
};

struct InputDataFormat {
  using type = std::variant<::InputDataFormat, AdmMetricNodalOptions>;
  static constexpr Options::String help =
      "The type of data stored in the 'InputH5Files'. Can be 'AdmMetricNodal' "
      "with additional options, 'MetricNodal', 'MetricModal', 'BondiNodal', or "
      "'BondiModal'.";
};

struct OutputH5File {
  using type = std::string;
  static constexpr Options::String help =
      "Name of output H5 file. A '.h5' extension will be added if needed.";
};

struct ExtractionRadius {
  using type = Options::Auto<double>;
  static constexpr Options::String help =
      "The radius of the spherical worldtube. "
      "If the 'InputH5File' supplied ends with '_RXXXX.h5' (where XXXX is the "
      "zero-padded extraction radius rounded to the nearest integer), then "
      "this option should be 'Auto'. If the extraction radius is not supplied "
      "in the 'InputH5File' name, then this option must be supplied. If the "
      "extraction radius is supplied in the 'InputH5File' name, and this "
      "option is specified, then this option will take precedence.";
};

struct FixSpecNormalization {
  using type = bool;
  static constexpr Options::String help =
      "Apply corrections associated with documented SpEC worldtube file "
      "errors. If you are using worldtube data from SpECTRE or from another "
      "NR code but in the SpECTRE format, then this option must be 'False'";
};

struct DescendingM {
  using type = bool;
  static constexpr Options::String help =
      "Whether the order of 'm' modes of the data for a given 'l' are stored "
      "in descending order (m goes from +l to -l) or not (m goes from -l to "
      "+l). This option must be 'False' when using worldtube data from SpECTRE "
      "or another NR code in the SpECTRE format. This should only be 'True' if "
      "your worldtube file is from the SpEC code.";
  static bool suggested_value() { return false; }
};

struct BufferDepth {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help =
      "Number of time steps to load during each call to the file-accessing "
      "routines. Higher values mean fewer, larger loads from file into RAM. "
      "Set to 'Auto' to use a default value (2000).";
};

struct LMaxFactor {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help =
      "The boundary computations will be performed at a resolution that is "
      "'LMaxFactor' times the input file LMax to avoid aliasing. Set to "
      "'Auto' to use a default value (2).";
};
}  // namespace OptionTags

using option_tags =
    tmpl::list<OptionTags::InputH5Files, OptionTags::InputDataFormat,
               OptionTags::OutputH5File, OptionTags::ExtractionRadius,
               OptionTags::FixSpecNormalization, OptionTags::DescendingM,
               OptionTags::BufferDepth, OptionTags::LMaxFactor>;
using OptionTuple = tuples::tagged_tuple_from_typelist<option_tags>;

namespace ReduceCceTags {
struct InputH5Files : db::SimpleTag {
  using type = std::vector<std::string>;
  using option_tags = tmpl::list<OptionTags::InputH5Files>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::variant<std::string, std::vector<std::string>>& option) {
    std::vector<std::string> result = std::visit(
        Overloader{[](const std::vector<std::string>& input) { return input; },
                   [](const std::string& input) { return std::vector{input}; }},
        option);
    for (std::string& filename : result) {
      if (not filename.ends_with(".h5")) {
        filename += ".h5";
      }
    }

    return result;
  }
};

struct InputDataFormat : db::SimpleTag {
 private:
  using option_type = std::variant<::InputDataFormat, AdmMetricNodalOptions>;

 public:
  using type = std::variant<::InputDataFormat, AdmOptions>;
  using option_tags = tmpl::list<OptionTags::InputDataFormat>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(option_type input_data_format) {
    type result{};
    if (std::holds_alternative<::InputDataFormat>(input_data_format)) {
      result = std::get<::InputDataFormat>(input_data_format);
    } else {
      result =
          std::get<AdmMetricNodalOptions>(input_data_format).adm_metric_nodal;
    }

    return result;
  }
};

struct OutputH5File : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::OutputH5File>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(std::string option) {
    if (not option.ends_with(".h5")) {
      option += ".h5";
    }
    return option;
  }
};

struct ExtractionRadius : db::SimpleTag {
  using type = std::optional<double>;
  using option_tags = tmpl::list<OptionTags::ExtractionRadius>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const std::optional<double>& option) {
    return option;
  }
};

struct FixSpecNormalization : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::FixSpecNormalization>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool option) { return option; }
};

struct DescendingM : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::DescendingM>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const bool option) { return option; }
};

struct BufferDepth : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::BufferDepth>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const std::optional<size_t>& option) {
    return option.value_or(2000);
  }
};

struct LMaxFactor : db::SimpleTag {
  using type = size_t;
  using option_tags = tmpl::list<OptionTags::BufferDepth>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const std::optional<size_t>& option) {
    return option.value_or(2);
  }
};
}  // namespace ReduceCceTags

using tags =
    tmpl::list<ReduceCceTags::InputH5Files, ReduceCceTags::InputDataFormat,
               ReduceCceTags::OutputH5File, ReduceCceTags::ExtractionRadius,
               ReduceCceTags::FixSpecNormalization, ReduceCceTags::DescendingM,
               ReduceCceTags::BufferDepth, ReduceCceTags::LMaxFactor>;
using TagsTuple = tuples::tagged_tuple_from_typelist<tags>;
}  // namespace

// Has to be outside the anon namespace
template <>
struct Options::create_from_yaml<InputDataFormat> {
  template <typename Metavariables>
  static InputDataFormat create(const Options::Option& options) {
    const auto ordering = options.parse_as<std::string>();
    if (ordering == "MetricNodal") {
      return InputDataFormat::MetricNodal;
    } else if (ordering == "MetricModal") {
      return InputDataFormat::MetricModal;
    } else if (ordering == "BondiNodal") {
      return InputDataFormat::BondiNodal;
    } else if (ordering == "BondiModal") {
      return InputDataFormat::BondiModal;
    }
    PARSE_ERROR(options.context(),
                "InputDataFormat must be 'AdmMetricNodal', 'MetricNodal', "
                "'MetricModal', 'BondiNodal', or 'BondiModal'");
  }
};

/*
 * This executable is used for converting the unnecessarily large SpEC worldtube
 * data format into a far smaller representation (roughly a factor of 4) just
 * storing the worldtube scalars that are required as input for CCE.
 */
int main(int argc, char** argv) {
  // Boost options for the input yaml and --help flag
  boost::program_options::options_description desc("Options");
  desc.add_options()("help,h,", "show this help message")(
      "input-file", boost::program_options::value<std::string>()->required(),
      "Name of YAML input file to use.");

  boost::program_options::variables_map vars;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vars);

  // Option parser for all the actual options
  Options::Parser<option_tags> parser{
      "This executable is used for converting the unnecessarily large metric "
      "worldtube data format into a smaller representation (roughly a factor "
      "of 4) just storing the worldtube scalars that are required as "
      "input for CCE."};

  // Help is a successful return
  if (vars.contains("help")) {
    Parallel::printf("%s\n%s", desc, parser.help());
    return 0;
  }

  // Not specifying an input file is an error
  if (not vars.contains("input-file")) {
    Parallel::printf("Missing input file. Pass '--input-file'");
    return 1;
  }

  // Wrap in try-catch to print nice errors and terminate gracefully
  try {
    const std::string input_yaml = vars["input-file"].as<std::string>();

    // Actually parse the yaml. This does a check if it exists.
    parser.parse_file(input_yaml, false);

    // First create option tags, and then actual tags.
    const OptionTuple options = parser.template apply<option_tags>(
        [](auto... args) { return OptionTuple(std::move(args)...); });
    const TagsTuple inputs =
        Parallel::create_from_options<void>(options, tags{});

    const auto& input_data_format =
        tuples::get<ReduceCceTags::InputDataFormat>(inputs);
    const InputDataFormat input_data_format_enum =
        std::holds_alternative<InputDataFormat>(input_data_format)
            ? std::get<InputDataFormat>(input_data_format)
            : InputDataFormat::AdmMetricNodal;
    const std::vector<std::string>& input_files =
        tuples::get<ReduceCceTags::InputH5Files>(inputs);
    const std::string& output_h5_file =
        tuples::get<ReduceCceTags::OutputH5File>(inputs);

    std::optional<std::string> temporary_combined_h5_file{};

    if (input_files.size() != 1) {
      // If the input format is BondiModal, then we don't actually have to do
      // any transformations, only combining H5 files. So the temporary file
      // name is just the output file
      if (input_data_format_enum == InputDataFormat::BondiModal) {
        temporary_combined_h5_file = output_h5_file;
      } else {
        // Otherwise we have to do a transformation so a temporary H5 file is
        // necessary. Name the file based on the current time so it doesn't
        // conflict with another h5 file
        const auto now = std::chrono::system_clock::now();
        const auto now_ns =
            std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
        const auto value = now_ns.time_since_epoch();
        temporary_combined_h5_file =
            "tmp_combined_" + std::to_string(value.count()) + ".h5";
      }

      // Now combine the h5 files into a single file
      h5::combine_h5_dat(input_files, temporary_combined_h5_file.value(),
                         Verbosity::Quiet);
    } else if (input_data_format_enum == InputDataFormat::BondiModal) {
      // Error here if the input data format is BondiModal since there's nothing
      // to do
      ERROR_NO_TRACE(
          "Only a single input H5 file was supplied and the input data "
          "format is BondiModal. This means that no combination needs to be "
          "done and running PreprocessCceWorldtube is unnecessary.");
    }

    if (tuples::get<ReduceCceTags::FixSpecNormalization>(inputs)) {
      if (input_data_format_enum != InputDataFormat::MetricModal) {
        ERROR_NO_TRACE(
            "The option FixSpecNormalization can only be 'true' when the input "
            "data format is MetricModal. Otherwise, it must be 'false'");
      }
      if (not tuples::get<ReduceCceTags::DescendingM>(inputs)) {
        ERROR_NO_TRACE(
            "The option FixSpecNormalization can only be 'true' if "
            "'DescendingM' is true as well.");
      }
    }

    const auto input_worldtube_filename = [&]() -> const std::string& {
      return temporary_combined_h5_file.has_value()
                 ? temporary_combined_h5_file.value()
                 : input_files[0];
    };

    const auto clean_temporary_file = [&temporary_combined_h5_file]() {
      if (temporary_combined_h5_file.has_value()) {
        file_system::rm(temporary_combined_h5_file.value(), false);
      }
    };

    switch (input_data_format_enum) {
      case InputDataFormat::BondiModal: {
        // Nothing to do here because this is the desired output format and the
        // H5 files were combined above
        return 0;
      }
      case InputDataFormat::BondiNodal: {
        bondi_nodal_to_bondi_modal(
            input_worldtube_filename(), output_h5_file,
            tuples::get<ReduceCceTags::BufferDepth>(inputs),
            tuples::get<ReduceCceTags::ExtractionRadius>(inputs));

        clean_temporary_file();
        return 0;
      }
      case InputDataFormat::MetricModal: {
        perform_cce_worldtube_reduction(
            input_worldtube_filename(), output_h5_file,
            tuples::get<ReduceCceTags::BufferDepth>(inputs),
            tuples::get<ReduceCceTags::LMaxFactor>(inputs),
            tuples::get<ReduceCceTags::ExtractionRadius>(inputs),
            tuples::get<ReduceCceTags::FixSpecNormalization>(inputs),
            tuples::get<ReduceCceTags::DescendingM>(inputs));

        clean_temporary_file();
        return 0;
      }
      case InputDataFormat::AdmMetricNodal: {
        [[fallthrough]];
      }
      case InputDataFormat::MetricNodal: {
        const std::optional<AdmOptions> adm_options =
            std::holds_alternative<AdmOptions>(input_data_format)
                ? std::optional{std::get<AdmOptions>(input_data_format)}
                : std::nullopt;
        metric_nodal_to_bondi_modal(
            input_worldtube_filename(), output_h5_file,
            tuples::get<ReduceCceTags::BufferDepth>(inputs),
            tuples::get<ReduceCceTags::ExtractionRadius>(inputs), adm_options);

        clean_temporary_file();
        return 0;
      }
      default:
        ERROR("Unknown input data format " << input_data_format_enum);
    }
  } catch (const std::exception& exception) {
    Parallel::printf("%s\n", exception.what());
    return 1;
  }
}
