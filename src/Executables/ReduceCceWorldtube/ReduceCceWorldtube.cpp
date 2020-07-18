// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <boost/program_options.hpp>
#include <cstddef>
#include <string>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/ReducedWorldtubeModeRecorder.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "Parallel/Exit.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// from a time-varies-fastest set of buffers provided by
// `SpecWorldtubeH5BufferUpdater` extract the set of coefficients for a
// particular time given by `buffer_time_offset` into the `time_span` size of
// buffer.
void slice_buffers_to_libsharp_modes(
    const gsl::not_null<Variables<Cce::cce_input_tags>*> coefficients_set,
    const Variables<Cce::cce_input_tags>& coefficients_buffers,
    const size_t time_span, const size_t buffer_time_offset, const size_t l_max,
    const size_t computation_l_max) noexcept {
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
             &computation_l_max, &time_span,
             &buffer_time_offset](auto tag_v) noexcept {
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
           &buffer_time_offset](auto tag_v) noexcept {
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
         &buffer_time_offset](auto tag_v) noexcept {
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
void perform_cce_worldtube_reduction(const std::string& input_file,
                                     const std::string& output_file,
                                     const size_t buffer_depth,
                                     const size_t l_max_factor) noexcept {
  Cce::SpecWorldtubeH5BufferUpdater buffer_updater{input_file};
  const size_t l_max = buffer_updater.get_l_max();
  // Perform the boundary computation to scalars at twice the input l_max to be
  // absolutely certain that there are no problems associated with aliasing.
  const size_t computation_l_max = l_max_factor * l_max;

  // we're not interpolating, this is just a reasonable number of rows to ingest
  // at a time.
  const size_t size_of_buffer = square(l_max + 1) * (buffer_depth);
  const DataVector& time_buffer = buffer_updater.get_time_buffer();

  Variables<Cce::cce_input_tags> coefficients_buffers{size_of_buffer};
  Variables<Cce::cce_input_tags> coefficients_set{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(computation_l_max)};

  using boundary_variables_tag =
      Tags::Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>>;

  auto boundary_data_box =
      db::create<db::AddSimpleTags<boundary_variables_tag>>(
          db::item_type<boundary_variables_tag>{
              Spectral::Swsh::number_of_swsh_collocation_points(
                  computation_l_max)});

  using reduced_boundary_tags =
      tmpl::list<Cce::Tags::BoundaryValue<Cce::Tags::BondiBeta>,
                 Cce::Tags::BoundaryValue<Cce::Tags::BondiU>,
                 Cce::Tags::BoundaryValue<Cce::Tags::BondiQ>,
                 Cce::Tags::BoundaryValue<Cce::Tags::BondiW>,
                 Cce::Tags::BoundaryValue<Cce::Tags::BondiJ>,
                 Cce::Tags::BoundaryValue<Cce::Tags::Dr<Cce::Tags::BondiJ>>,
                 Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiJ>>,
                 Cce::Tags::BoundaryValue<Cce::Tags::BondiR>,
                 Cce::Tags::BoundaryValue<Cce::Tags::Du<Cce::Tags::BondiR>>>;

  size_t time_span_start = 0;
  size_t time_span_end = 0;
  Cce::ReducedWorldtubeModeRecorder recorder{output_file};

  ComplexModalVector output_goldberg_mode_buffer{square(computation_l_max + 1)};
  ComplexModalVector output_libsharp_mode_buffer{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(computation_l_max)};

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

    if (buffer_updater.radial_derivatives_need_renormalization()) {
      Cce::create_bondi_boundary_data_from_unnormalized_spec_modes(
          make_not_null(&boundary_data_box),
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
          make_not_null(&boundary_data_box),
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
    tmpl::for_each<reduced_boundary_tags>(
        [&recorder, &boundary_data_box, &output_goldberg_mode_buffer,
         &output_libsharp_mode_buffer, &l_max, &computation_l_max,
         &time](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          SpinWeighted<ComplexModalVector, db::item_type<tag>::type::spin>
              spin_weighted_libsharp_view;
          spin_weighted_libsharp_view.set_data_ref(
              output_libsharp_mode_buffer.data(),
              output_libsharp_mode_buffer.size());
          Spectral::Swsh::swsh_transform(
              computation_l_max, 1, make_not_null(&spin_weighted_libsharp_view),
              get(db::get<tag>(boundary_data_box)));
          SpinWeighted<ComplexModalVector, db::item_type<tag>::type::spin>
              spin_weighted_goldberg_view;
          spin_weighted_goldberg_view.set_data_ref(
              output_goldberg_mode_buffer.data(),
              output_goldberg_mode_buffer.size());
          Spectral::Swsh::libsharp_to_goldberg_modes(
              make_not_null(&spin_weighted_goldberg_view),
              spin_weighted_libsharp_view, computation_l_max);

          // The goldberg format type is in strictly increasing l modes, so to
          // reduce to a smaller l_max, we can just take the first (l_max + 1)^2
          // values.
          ComplexModalVector reduced_goldberg_view{
              output_goldberg_mode_buffer.data(), square(l_max + 1)};
          recorder.append_worldtube_mode_data(
              "/" + Cce::dataset_label_for_tag<tag>(), time,
              reduced_goldberg_view, l_max,
              db::item_type<tag>::type::spin == 0);
        });
  }
  Parallel::printf("\n");
}

/*
 * This executable is used for converting the unnecessarily large SpEC worldtube
 * data format into a far smaller representation (roughly a factor of 4) just
 * storing the worldtube scalars that are required as input for CCE.
 */
int main(int argc, char** argv) {
  boost::program_options::positional_options_description pos_desc;
  pos_desc.add("old_spec_cce_file", 1).add("output_file", 1);

  boost::program_options::options_description desc("Options");
  desc.add_options()("help", "show this help message")(
      "input_file", boost::program_options::value<std::string>()->required(),
      "name of old CCE data file")(
      "output_file", boost::program_options::value<std::string>()->required(),
      "output filename")(
      "buffer_depth",
      boost::program_options::value<size_t>()->default_value(2000),
      "number of time steps to load during each call to the file-accessing "
      "routines. Higher values mean fewer, larger loads from file into RAM.")(
      "lmax_factor", boost::program_options::value<size_t>()->default_value(2),
      "the boundary computations will be performed at a resolution that is "
      "lmax_factor times the input file lmax to avoid aliasing");

  boost::program_options::variables_map vars;

  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .positional(pos_desc)
          .options(desc)
          .run(),
      vars);

  if (vars.count("help") != 0u) {
    Parallel::printf("%s\n", desc);
    Parallel::exit();
  }

  perform_cce_worldtube_reduction(vars["input_file"].as<std::string>(),
                                  vars["output_file"].as<std::string>(),
                                  vars["buffer_depth"].as<size_t>(),
                                  vars["lmax_factor"].as<size_t>());
}
