// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <exception>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Options/Auto.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Parallel/CreateFromOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

namespace {
// from a time-varies-fastest set of buffers provided by
// `MetricWorldtubeH5BufferUpdater` extract the set of coefficients for a
// particular time given by `buffer_time_offset` into the `time_span` size of
// buffer.
void slice_buffers_to_libsharp_modes(
    const gsl::not_null<Variables<Cce::cce_metric_input_tags>*>
        coefficients_set,
    const Variables<Cce::cce_metric_input_tags>& coefficients_buffers,
    const size_t time_span, const size_t buffer_time_offset, const size_t l_max,
    const size_t computation_l_max) {
  SpinWeighted<ComplexModalVector, 0> spin_weighted_buffer;

  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(computation_l_max)) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        tmpl::for_each<
            tmpl::list<Cce::Tags::detail::SpatialMetric,
                       Cce::Tags::detail::Dr<Cce::Tags::detail::SpatialMetric>,
                       Tags::dt<Cce::Tags::detail::SpatialMetric>>>(
            [&i, &j, &libsharp_mode, &spin_weighted_buffer,
             &coefficients_buffers, &coefficients_set, &l_max,
             &computation_l_max, &time_span, &buffer_time_offset](auto tag_v) {
              using tag = typename decltype(tag_v)::type;
              spin_weighted_buffer.set_data_ref(
                  get<tag>(*coefficients_set).get(i, j).data(),
                  Spectral::Swsh::size_of_libsharp_coefficient_vector(
                      computation_l_max));
              if (libsharp_mode.l > l_max) {
                Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                    libsharp_mode, make_not_null(&spin_weighted_buffer), 0, 0.0,
                    0.0);

              } else {
                Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                    libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                    get<tag>(coefficients_buffers)
                        .get(i, j)[time_span *
                                       Spectral::Swsh::goldberg_mode_index(
                                           l_max, libsharp_mode.l,
                                           static_cast<int>(libsharp_mode.m)) +
                                   buffer_time_offset],
                    get<tag>(coefficients_buffers)
                        .get(i, j)[time_span *
                                       Spectral::Swsh::goldberg_mode_index(
                                           l_max, libsharp_mode.l,
                                           -static_cast<int>(libsharp_mode.m)) +
                                   buffer_time_offset]);
              }
            });
      }
      tmpl::for_each<tmpl::list<Cce::Tags::detail::Shift,
                                Cce::Tags::detail::Dr<Cce::Tags::detail::Shift>,
                                Tags::dt<Cce::Tags::detail::Shift>>>(
          [&i, &libsharp_mode, &spin_weighted_buffer, &coefficients_buffers,
           &coefficients_set, &l_max, &computation_l_max, &time_span,
           &buffer_time_offset](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            spin_weighted_buffer.set_data_ref(
                get<tag>(*coefficients_set).get(i).data(),
                Spectral::Swsh::size_of_libsharp_coefficient_vector(
                    computation_l_max));

            if (libsharp_mode.l > l_max) {
              Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                  libsharp_mode, make_not_null(&spin_weighted_buffer), 0, 0.0,
                  0.0);

            } else {
              Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                  libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                  get<tag>(coefficients_buffers)
                      .get(i)[time_span *
                                  Spectral::Swsh::goldberg_mode_index(
                                      l_max, libsharp_mode.l,
                                      static_cast<int>(libsharp_mode.m)) +
                              buffer_time_offset],
                  get<tag>(coefficients_buffers)
                      .get(i)[time_span *
                                  Spectral::Swsh::goldberg_mode_index(
                                      l_max, libsharp_mode.l,
                                      -static_cast<int>(libsharp_mode.m)) +
                              buffer_time_offset]);
            }
          });
    }
    tmpl::for_each<tmpl::list<Cce::Tags::detail::Lapse,
                              Cce::Tags::detail::Dr<Cce::Tags::detail::Lapse>,
                              Tags::dt<Cce::Tags::detail::Lapse>>>(
        [&libsharp_mode, &spin_weighted_buffer, &coefficients_buffers,
         &coefficients_set, &l_max, &computation_l_max, &time_span,
         &buffer_time_offset](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          spin_weighted_buffer.set_data_ref(
              get(get<tag>(*coefficients_set)).data(),
              Spectral::Swsh::size_of_libsharp_coefficient_vector(
                  computation_l_max));

          if (libsharp_mode.l > l_max) {
            Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                libsharp_mode, make_not_null(&spin_weighted_buffer), 0, 0.0,
                0.0);

          } else {
            Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                get(get<tag>(coefficients_buffers))
                    [time_span * Spectral::Swsh::goldberg_mode_index(
                                     l_max, libsharp_mode.l,
                                     static_cast<int>(libsharp_mode.m)) +
                     buffer_time_offset],
                get(get<tag>(coefficients_buffers))
                    [time_span * Spectral::Swsh::goldberg_mode_index(
                                     l_max, libsharp_mode.l,
                                     -static_cast<int>(libsharp_mode.m)) +
                     buffer_time_offset]);
          }
        });
  }
}

// read in the data from a (previously standard) SpEC worldtube file
// `input_file`, perform the boundary computation, and dump the (considerably
// smaller) dataset associated with the spin-weighted scalars to `output_file`.
void perform_cce_worldtube_reduction(
    const std::string& input_file, const std::string& output_file,
    const size_t buffer_depth, const size_t l_max_factor,
    const bool fix_spec_normalization = false) {
  Cce::MetricWorldtubeH5BufferUpdater buffer_updater{input_file};
  const size_t l_max = buffer_updater.get_l_max();
  // Perform the boundary computation to scalars at twice the input l_max to be
  // absolutely certain that there are no problems associated with aliasing.
  const size_t computation_l_max = l_max_factor * l_max;

  // we're not interpolating, this is just a reasonable number of rows to ingest
  // at a time.
  const size_t size_of_buffer = square(l_max + 1) * (buffer_depth);
  const DataVector& time_buffer = buffer_updater.get_time_buffer();

  Variables<Cce::cce_metric_input_tags> coefficients_buffers{size_of_buffer};
  Variables<Cce::cce_metric_input_tags> coefficients_set{
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
    Parallel::printf("reducing data at time : %f / %f \r", time,
                     time_buffer[time_buffer.size() - 1]);
    buffer_updater.update_buffers_for_time(
        make_not_null(&coefficients_buffers), make_not_null(&time_span_start),
        make_not_null(&time_span_end), time, l_max, 0, buffer_depth);

    slice_buffers_to_libsharp_modes(
        make_not_null(&coefficients_set), coefficients_buffers,
        time_span_end - time_span_start, i - time_span_start, l_max,
        computation_l_max);

    if (not buffer_updater.has_version_history() and fix_spec_normalization) {
      Cce::create_bondi_boundary_data_from_unnormalized_spec_modes(
          make_not_null(&boundary_data_variables),
          get<Cce::Tags::detail::SpatialMetric>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::SpatialMetric>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::SpatialMetric>>(
              coefficients_set),
          get<Cce::Tags::detail::Shift>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::Shift>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::Shift>>(
              coefficients_set),
          get<Cce::Tags::detail::Lapse>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::Lapse>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::Lapse>>(
              coefficients_set),
          buffer_updater.get_extraction_radius(), computation_l_max);
    } else {
      Cce::create_bondi_boundary_data(
          make_not_null(&boundary_data_variables),
          get<Cce::Tags::detail::SpatialMetric>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::SpatialMetric>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::SpatialMetric>>(
              coefficients_set),
          get<Cce::Tags::detail::Shift>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::Shift>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::Shift>>(
              coefficients_set),
          get<Cce::Tags::detail::Lapse>(coefficients_set),
          get<Tags::dt<Cce::Tags::detail::Lapse>>(coefficients_set),
          get<Cce::Tags::detail::Dr<Cce::Tags::detail::Lapse>>(
              coefficients_set),
          buffer_updater.get_extraction_radius(), computation_l_max);
    }
    // loop over the tags that we want to dump.
    tmpl::for_each<Cce::Tags::worldtube_boundary_tags_for_writing<>>(
        [&recorder, &boundary_data_variables, &l_max, &time](auto tag_v) {
          using tag = typename decltype(tag_v)::type;

          const ComplexDataVector& nodal_data =
              get(get<tag>(boundary_data_variables)).data();
          // The goldberg format type is in strictly increasing l modes, so to
          // reduce to a smaller l_max, we can just take the first (l_max + 1)^2
          // values.
          const ComplexDataVector nodal_data_view{
              make_not_null(
                  const_cast<ComplexDataVector&>(nodal_data).data()),  // NOLINT
              square(l_max + 1)};

          recorder.append_modal_data<tag::tag::type::type::spin>(
              Cce::dataset_label_for_tag<typename tag::tag>(), time,
              nodal_data_view);
        });
  }
  Parallel::printf("\n");
}

namespace OptionTags {
struct InputH5File {
  using type = std::string;
  static constexpr Options::String help =
      "Name of the H5 worldtube file. A '.h5' extension will be added if "
      "needed.";
};

struct OutputH5File {
  using type = std::string;
  static constexpr Options::String help =
      "Name of output H5 file. A '.h5' extension will be added if needed.";
};

struct FixSpecNormalization {
  using type = bool;
  static constexpr Options::String help =
      "Apply corrections associated with documented SpEC worldtube file "
      "errors. If you are using worldtube data from SpECTRE or from another NR "
      "code but in the SpECTRE format, then this option must be 'False'";
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
      "'LMaxFactor' times the input file LMax to avoid aliasing. Set to 'Auto' "
      "to use a default value (2).";
};
}  // namespace OptionTags

using option_tags =
    tmpl::list<OptionTags::InputH5File, OptionTags::OutputH5File,
               OptionTags::FixSpecNormalization, OptionTags::BufferDepth,
               OptionTags::LMaxFactor>;
using OptionTuple = tuples::tagged_tuple_from_typelist<option_tags>;

namespace ReduceCceTags {
struct InputH5File : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<OptionTags::InputH5File>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(std::string option) {
    if (not option.ends_with(".h5")) {
      option += ".h5";
    }
    return option;
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

struct FixSpecNormalization : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::FixSpecNormalization>;
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

using tags = tmpl::list<ReduceCceTags::InputH5File, ReduceCceTags::OutputH5File,
                        ReduceCceTags::FixSpecNormalization,
                        ReduceCceTags::BufferDepth, ReduceCceTags::LMaxFactor>;
using TagsTuple = tuples::tagged_tuple_from_typelist<tags>;
}  // namespace

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
    parser.parse_file(input_yaml);

    // First create option tags, and then actual tags.
    const OptionTuple options = parser.template apply<option_tags>(
        [](auto... args) { return OptionTuple(std::move(args)...); });
    const TagsTuple inputs =
        Parallel::create_from_options<void>(options, tags{});

    // Do the reduction
    perform_cce_worldtube_reduction(
        tuples::get<ReduceCceTags::InputH5File>(inputs),
        tuples::get<ReduceCceTags::OutputH5File>(inputs),
        tuples::get<ReduceCceTags::BufferDepth>(inputs),
        tuples::get<ReduceCceTags::BufferDepth>(inputs),
        tuples::get<ReduceCceTags::FixSpecNormalization>(inputs));
  } catch (const std::exception& exception) {
    Parallel::printf("%s\n", exception.what());
    return 1;
  }
}
