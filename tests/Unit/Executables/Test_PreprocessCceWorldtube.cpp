// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <brigand/brigand.hpp>
#include <cstdlib>
#include <fstream>
#include <optional>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Math.hpp"

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#endif  // defined(__GNUC__) and not defined(__clang__)
// Need this for linking, but it doesn't do anything
extern "C" void CkRegisterMainModule() {}
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) and not defined(__clang__)

namespace {
constexpr size_t number_of_times = 30;

double compute_time(const double target_time, const size_t time_index) {
  // This formula matches the one in BoundaryTestHelpers.hpp which we use to
  // write some of the worldtube data to disk
  return 0.1 * static_cast<double>(time_index) + target_time - 1.5;
}

using modal_tags = Cce::Tags::worldtube_boundary_tags_for_writing<
    Spectral::Swsh::Tags::SwshTransform>;
using ExpectedDataType = std::vector<Variables<modal_tags>>;

template <typename Solution>
ExpectedDataType create_expected_data(const size_t l_max,
                                      const double target_time,
                                      const double extraction_radius,
                                      const Solution& solution,
                                      const double amplitude,
                                      const double frequency) {
  const size_t computation_l_max = 3 * l_max;
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(computation_l_max);
  Variables<Cce::Tags::characteristic_worldtube_boundary_tags<
      Cce::Tags::BoundaryValue>>
      boundary_data_variables{number_of_angular_points};

  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(computation_l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};

  const size_t computation_modal_size = square(computation_l_max + 1);
  ComplexModalVector computation_goldberg_mode_buffer{computation_modal_size};

  const size_t modal_size = square(l_max + 1);
  std::vector<Variables<modal_tags>> result{number_of_times};
  for (size_t i = 0; i < number_of_times; i++) {
    result[i] = Variables<modal_tags>{modal_size};
  }

  for (size_t t = 0; t < number_of_times; ++t) {
    const double time = compute_time(target_time, t);
    // Create fake metric nodal data
    Cce::TestHelpers::create_fake_time_varying_data(
        make_not_null(&spatial_metric_coefficients),
        make_not_null(&dt_spatial_metric_coefficients),
        make_not_null(&dr_spatial_metric_coefficients),
        make_not_null(&shift_coefficients),
        make_not_null(&dt_shift_coefficients),
        make_not_null(&dr_shift_coefficients),
        make_not_null(&lapse_coefficients),
        make_not_null(&dt_lapse_coefficients),
        make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
        amplitude, frequency, time, computation_l_max, false);

    // Convert to Bondi nodal
    Cce::create_bondi_boundary_data(
        make_not_null(&boundary_data_variables), spatial_metric_coefficients,
        dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
        shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
        lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
        extraction_radius, computation_l_max);

    // Convert to Bondi modal
    tmpl::for_each<modal_tags>([&](auto tag_v) {
      using wrapped_tag = tmpl::type_from<decltype(tag_v)>;
      using tag = typename wrapped_tag::tag;
      constexpr int Spin = tag::type::type::spin;

      SpinWeighted<ComplexDataVector, Spin> nodal_data_view;
      nodal_data_view.set_data_ref(
          make_not_null(&const_cast<ComplexDataVector&>(  // NOLINT
              get(get<Cce::Tags::BoundaryValue<tag>>(boundary_data_variables))
                  .data())));
      SpinWeighted<ComplexModalVector, Spin> goldberg_modes;
      goldberg_modes.set_data_ref(computation_goldberg_mode_buffer.data(),
                                  computation_modal_size);

      // First transform to coefficients using swsh_transform, and then convert
      // libsharp coefficients into modes
      Spectral::Swsh::libsharp_to_goldberg_modes(
          make_not_null(&goldberg_modes),
          Spectral::Swsh::swsh_transform(computation_l_max, 1, nodal_data_view),
          computation_l_max);

      // Restrict the values back to the correct modal size from the computation
      // modal size
      ComplexModalVector goldberg_mode_view;
      goldberg_mode_view.set_data_ref(goldberg_modes.data().data(), modal_size);

      get(get<wrapped_tag>(result[t])).data() = goldberg_mode_view;
    });
  }

  return result;
}

// Accuracy to which we check the data
constexpr double epsilon = 1.e-12;

void check_expected_data(const std::string& output_filename, const size_t l_max,
                         const ExpectedDataType& expected_data,
                         const ExpectedDataType& second_expected_data,
                         const double target_time,
                         const double second_target_time,
                         const bool check_second_expected_data) {
  const size_t expected_number_of_times =
      check_second_expected_data ? 2 * number_of_times : number_of_times;
  const h5::H5File<h5::AccessType::ReadOnly> output_file{output_filename};

  CAPTURE_FOR_ERROR(output_filename);
  CAPTURE_FOR_ERROR(epsilon);

  tmpl::for_each<modal_tags>([&](auto tag_v) {
    using wrapped_tag = tmpl::type_from<decltype(tag_v)>;
    using tag = typename wrapped_tag::tag;
    constexpr int Spin = tag::type::type::spin;
    constexpr bool is_real = Spin == 0;

    const std::string dataset_name = Cce::dataset_label_for_tag<tag>();
    const auto& bondi_dat_subfile = output_file.get<h5::Dat>(dataset_name);
    CAPTURE_FOR_ERROR(dataset_name);
    const auto bondi_data =
        bondi_dat_subfile.get_data<std::vector<std::vector<double>>>();

    SPECTRE_PARALLEL_REQUIRE(bondi_data.size() == expected_number_of_times);

    const auto check_data_for_target_time = [&](const double local_target_time,
                                                const auto& local_expected_data,
                                                const size_t offset) {
      for (size_t t = 0; t < number_of_times; t++) {
        const double exptected_time = compute_time(local_target_time, t);
        const auto& expected_bondi_var =
            get(get<wrapped_tag>(local_expected_data[t])).data();
        const std::vector<double>& bondi_var = bondi_data[offset + t];

        CAPTURE_FOR_ERROR(bondi_var[0]);
        CAPTURE_FOR_ERROR(t);
        CAPTURE_FOR_ERROR(exptected_time);
        SPECTRE_PARALLEL_REQUIRE(bondi_var[0] == exptected_time);

        (void)expected_bondi_var;

        CAPTURE_FOR_ERROR(expected_bondi_var);
        CAPTURE_FOR_ERROR(bondi_var);

        for (int l = 0; l <= static_cast<int>(l_max); l++) {
          for (int m = (is_real ? 0 : -l); m <= l; m++) {
            const size_t goldberg_index = Spectral::Swsh::goldberg_mode_index(
                l_max, static_cast<size_t>(l), m);

            CAPTURE_FOR_ERROR(l);
            CAPTURE_FOR_ERROR(m);
            CAPTURE_FOR_ERROR(goldberg_index);
            CAPTURE_FOR_ERROR(expected_bondi_var[goldberg_index]);
            std::complex<double> written_mode{};
            size_t matrix_index = 0;
            if (is_real) {
              if (m == 0) {
                matrix_index = static_cast<size_t>(square(l));  // NOLINT
                written_mode =
                    std::complex<double>{bondi_var[1 + matrix_index], 0.0};
              } else {
                matrix_index =
                    static_cast<size_t>(square(l) + 2 * abs(m));  // NOLINT
                written_mode =
                    ((m > 0 or abs(m) % 2 == 0) ? 1.0 : -1.0) *
                    std::complex<double>{bondi_var[1 + matrix_index - 1],
                                         sgn(m) * bondi_var[1 + matrix_index]};
              }
            } else {
              matrix_index = goldberg_index;
              written_mode = std::complex<double>{
                  bondi_var[1 + 2 * matrix_index],
                  bondi_var[1 + 2 * matrix_index + 1],
              };
            }
            CAPTURE_FOR_ERROR(matrix_index);
            CAPTURE_FOR_ERROR(written_mode);
            SPECTRE_PARALLEL_REQUIRE(equal_within_roundoff(
                expected_bondi_var[goldberg_index], written_mode, epsilon));
          }
        }
      }
    };

    check_data_for_target_time(target_time, expected_data, 0);

    if (check_second_expected_data) {
      check_data_for_target_time(second_target_time, second_expected_data,
                                 number_of_times);
    }

    output_file.close_current_object();
  });
}

void write_input_file(const std::string& input_data_format,
                      const std::string& input_file_name,
                      const std::vector<std::string>& input_worldtube_filenames,
                      const std::string& output_filename,
                      const std::optional<double>& worldtube_radius,
                      const bool descending_m) {
  std::string input_file =
      "# Distributed under the MIT License.\n"
      "# See LICENSE.txt for details.\n"
      "\n"
      "InputH5File: ";

  if (input_worldtube_filenames.size() > 1) {
    input_file += "[";
  }

  for (size_t i = 0; i < input_worldtube_filenames.size(); i++) {
    input_file += input_worldtube_filenames[i];
    if (i != input_worldtube_filenames.size() - 1) {
      input_file += ", ";
    }
  }

  if (input_worldtube_filenames.size() > 1) {
    input_file += "]";
  }

  input_file += "\nOutputH5File: " + output_filename + "\n";
  input_file += "InputDataFormat: " + input_data_format + "\n";
  input_file +=
      "ExtractionRadius: " +
      (worldtube_radius.has_value() ? std::to_string(worldtube_radius.value())
                                    : "Auto") +
      "\n";
  input_file += "DescendingM: "s + (descending_m ? "True\n" : "False\n");
  input_file +=
      "FixSpecNormalization: False\n"
      "BufferDepth: Auto\n"
      "LMaxFactor: 3\n";

  std::ofstream yaml_file(input_file_name);
  yaml_file << input_file;
  yaml_file.close();
}
}  // namespace

int main() {
  const size_t l_max = 16;
  const double target_time = 20.0;
  const double second_target_time = target_time + 20.0;
  const double worldtube_radius = 123.0;
  // These are just to create fake data
  const double frequency = 0.01;
  const double amplitude = 0.01;

  const double mass = 3.5;
  const std::array<double, 3> spin{-0.3, -0.2, 0.1};
  const std::array<double, 3> center{0.0, 0.0, 0.0};
  const gr::Solutions::KerrSchild solution{mass, spin, center};

  // Input worldtube H5 filenames
  // Some have the worldtube radius and some don't to test the ExtractionRadius
  // option
  const std::string metric_modal_input_worldtube_filename{
      "Test_InputMetricModal_R0123.h5"};
  const std::string metric_modal_spec_input_worldtube_filename{
      "Test_InputMetricModalSpec_R0123.h5"};
  const std::string metric_nodal_1_input_worldtube_filename{
      "Test_InputMetricNodal_1.h5"};
  const std::string metric_nodal_2_input_worldtube_filename{
      "Test_InputMetricNodal_2.h5"};
  const std::string bondi_modal_1_input_worldtube_filename{
      "Test_InputBondiModal_1_R0123.h5"};
  const std::string bondi_modal_2_input_worldtube_filename{
      "Test_InputBondiModal_2_R0123.h5"};
  const std::string bondi_nodal_input_worldtube_filename{
      "Test_InputBondiNodal_R0123.h5"};

  // Output worldtube H5 filenames
  const std::string metric_modal_output_worldtube_filename{
      "Test_OutputMetricModal_R0123.h5"};
  const std::string metric_modal_spec_output_worldtube_filename{
      "Test_OutputMetricModalSpec_R0123.h5"};
  const std::string metric_nodal_output_worldtube_filename{
      "Test_OutputMetricNodal.h5"};
  const std::string bondi_modal_output_worldtube_filename{
      "Test_OutputBondiModal_R0123.h5"};
  const std::string bondi_nodal_output_worldtube_filename{
      "Test_OutputBondiNodal_R0123.h5"};

  // Write metric data
  Cce::TestHelpers::write_test_file<ComplexModalVector, false>(
      solution, metric_modal_input_worldtube_filename, target_time,
      worldtube_radius, frequency, amplitude, l_max, false);
  Cce::TestHelpers::write_test_file<ComplexModalVector, false>(
      solution, metric_modal_spec_input_worldtube_filename, target_time,
      worldtube_radius, frequency, amplitude, l_max, true);
  Cce::TestHelpers::write_test_file<DataVector, false>(
      solution, metric_nodal_1_input_worldtube_filename, target_time,
      worldtube_radius, frequency, amplitude, l_max, false);
  Cce::TestHelpers::write_test_file<DataVector, false>(
      solution, metric_nodal_2_input_worldtube_filename, second_target_time,
      worldtube_radius, frequency, amplitude, l_max, false);

  // Write bondi data
  Cce::TestHelpers::write_test_file<ComplexModalVector, true>(
      solution, bondi_modal_1_input_worldtube_filename, target_time,
      worldtube_radius, frequency, amplitude, l_max, false);
  Cce::TestHelpers::write_test_file<ComplexModalVector, true>(
      solution, bondi_modal_2_input_worldtube_filename, second_target_time,
      worldtube_radius, frequency, amplitude, l_max, false);
  Cce::TestHelpers::write_test_file<DataVector, true>(
      solution, bondi_nodal_input_worldtube_filename, target_time,
      worldtube_radius, frequency, amplitude, l_max, false);

  // Write input file
  write_input_file("MetricModal", "MetricModal.yaml",
                   {metric_modal_input_worldtube_filename},
                   metric_modal_output_worldtube_filename, std::nullopt, false);
  write_input_file("MetricModal", "MetricModalSpec.yaml",
                   {metric_modal_spec_input_worldtube_filename},
                   metric_modal_spec_output_worldtube_filename, std::nullopt,
                   true);
  write_input_file("MetricNodal", "MetricNodal.yaml",
                   {metric_nodal_1_input_worldtube_filename,
                    metric_nodal_2_input_worldtube_filename},
                   metric_nodal_output_worldtube_filename, {worldtube_radius},
                   false);
  write_input_file("BondiModal", "BondiModal.yaml",
                   {bondi_modal_1_input_worldtube_filename,
                    bondi_modal_2_input_worldtube_filename},
                   bondi_modal_output_worldtube_filename, std::nullopt, false);
  write_input_file(
      "BondiNodal", "BondiNodal.yaml", {bondi_nodal_input_worldtube_filename},
      bondi_nodal_output_worldtube_filename, {worldtube_radius}, false);

// Get path to executable with a macro set in CMakeLists.txt
#ifdef BINDIR
  std::string executable = BINDIR;
#else
  std::string executable = "nothing";
  ERROR(
      "BINDIR preprocessor macro not set from CMake. Something is wrong with "
      "the build system.");
#endif
  if (not executable.ends_with("/")) {
    executable += "/";
  }
  executable += "bin/PreprocessCceWorldtube";

  const auto call_preprocess_cce_worldtube =
      [&](const std::string& input_file_name) {
        const std::string to_execute = executable + " --input-file " +
                                       input_file_name + ".yaml > " +
                                       input_file_name + ".out 2>&1";

        CAPTURE_FOR_ERROR(input_file_name);
        CAPTURE_FOR_ERROR(to_execute);
        const int exit_code = std::system(to_execute.c_str());  // NOLINT

        SPECTRE_PARALLEL_REQUIRE(exit_code == 0);
        (void)exit_code;
      };

  // Call PreprocessCceWorldtube in a shell
  call_preprocess_cce_worldtube("MetricModal");
  call_preprocess_cce_worldtube("MetricModalSpec");
  call_preprocess_cce_worldtube("MetricNodal");
  call_preprocess_cce_worldtube("BondiModal");
  call_preprocess_cce_worldtube("BondiNodal");

  // Create the expected bondi modal data
  const auto expected_data = create_expected_data(
      l_max, target_time, worldtube_radius, solution, amplitude, frequency);
  const auto second_expected_data =
      create_expected_data(l_max, second_target_time, worldtube_radius,
                           solution, amplitude, frequency);

  // Check that the expected bondi modal data is what was written in the output
  // files for the different InputDataFormats
  check_expected_data(metric_modal_output_worldtube_filename, l_max,
                      expected_data, second_expected_data, target_time,
                      second_target_time, false);
  check_expected_data(metric_nodal_output_worldtube_filename, l_max,
                      expected_data, second_expected_data, target_time,
                      second_target_time, true);
  check_expected_data(bondi_modal_output_worldtube_filename, l_max,
                      expected_data, second_expected_data, target_time,
                      second_target_time, true);
  check_expected_data(bondi_nodal_output_worldtube_filename, l_max,
                      expected_data, second_expected_data, target_time,
                      second_target_time, false);

  return 0;
}
