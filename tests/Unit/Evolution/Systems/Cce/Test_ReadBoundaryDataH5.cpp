// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "tests/Unit/Evolution/Systems/Cce/WriteToWorldtubeH5.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {

namespace {
template <typename AnalyticSolution>
void create_fake_time_varying_modal_data(
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        dt_spatial_metric_coefficients,
    const gsl::not_null<tnsr::ii<ComplexModalVector, 3>*>
        dr_spatial_metric_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> shift_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> dt_shift_coefficients,
    const gsl::not_null<tnsr::I<ComplexModalVector, 3>*> dr_shift_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> lapse_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> dt_lapse_coefficients,
    const gsl::not_null<Scalar<ComplexModalVector>*> dr_lapse_coefficients,
    const AnalyticSolution& solution, const double extraction_radius,
    const double amplitude, const double frequency, const double time,
    const size_t l_max, const bool convert_to_goldberg = true) noexcept {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // create the vector of collocation points that we want to interpolate to

  tnsr::I<DataVector, 3> collocation_points{number_of_angular_points};
  const auto& collocation = Spectral::Swsh::cached_collocation_metadata<
      Spectral::Swsh::ComplexRepresentation::Interleaved>(l_max);
  for (const auto& collocation_point : collocation) {
    get<0>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * cos(collocation_point.phi);
    get<1>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        sin(collocation_point.theta) * sin(collocation_point.phi);
    get<2>(collocation_points)[collocation_point.offset] =
        extraction_radius * (1.0 + amplitude * sin(frequency * time)) *
        cos(collocation_point.theta);
  }

  const auto kerr_schild_variables = solution.variables(
      collocation_points, 0.0, gr::Solutions::KerrSchild::tags<DataVector>{});

  const Scalar<DataVector>& lapse =
      get<gr::Tags::Lapse<DataVector>>(kerr_schild_variables);
  const Scalar<DataVector>& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(kerr_schild_variables);
  const auto& d_lapse = get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(
      kerr_schild_variables);

  const auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
      kerr_schild_variables);
  const auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          kerr_schild_variables);
  const auto& d_shift = get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(
      kerr_schild_variables);

  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          kerr_schild_variables);
  const auto& dt_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      kerr_schild_variables);
  const auto& d_spatial_metric =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(
          kerr_schild_variables);

  Scalar<DataVector> dr_lapse{number_of_angular_points};
  get(dr_lapse) = (get<0>(collocation_points) * get<0>(d_lapse) +
                   get<1>(collocation_points) * get<1>(d_lapse) +
                   get<2>(collocation_points) * get<2>(d_lapse)) /
                  extraction_radius;
  tnsr::I<DataVector, 3> dr_shift{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    dr_shift.get(i) = (get<0>(collocation_points) * d_shift.get(0, i) +
                       get<1>(collocation_points) * d_shift.get(1, i) +
                       get<2>(collocation_points) * d_shift.get(2, i)) /
                      extraction_radius;
  }
  tnsr::ii<DataVector, 3> dr_spatial_metric{number_of_angular_points};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dr_spatial_metric.get(i, j) =
          (get<0>(collocation_points) * d_spatial_metric.get(0, i, j) +
           get<1>(collocation_points) * d_spatial_metric.get(1, i, j) +
           get<2>(collocation_points) * d_spatial_metric.get(2, i, j)) /
          extraction_radius;
    }
  }

  if (convert_to_goldberg) {
    *lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(lapse, l_max);
    *dt_lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_lapse, l_max);
    *dr_lapse_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_lapse, l_max);

    *shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(shift, l_max);
    *dt_shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_shift, l_max);
    *dr_shift_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_shift, l_max);

    *spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(spatial_metric, l_max);
    *dt_spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dt_spatial_metric, l_max);
    *dr_spatial_metric_coefficients =
        TestHelpers::tensor_to_goldberg_coefficients(dr_spatial_metric, l_max);
  } else {
    *lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(lapse, l_max);
    *dt_lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_lapse, l_max);
    *dr_lapse_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_lapse, l_max);

    *shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(shift, l_max);
    *dt_shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_shift, l_max);
    *dr_shift_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_shift, l_max);

    *spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(spatial_metric, l_max);
    *dt_spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dt_spatial_metric, l_max);
    *dr_spatial_metric_coefficients =
        TestHelpers::tensor_to_libsharp_coefficients(dr_spatial_metric, l_max);
  }
}

template <typename AnalyticSolution>
class DummyBufferUpdater : public WorldtubeBufferUpdater {
 public:
  DummyBufferUpdater(DataVector time_buffer, const AnalyticSolution& solution,
                     const double extraction_radius,
                     const double coordinate_amplitude,
                     const double coordinate_frequency,
                     const size_t l_max) noexcept
      : time_buffer_{std::move(time_buffer)},
        solution_{solution},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max} {}

  WRAPPED_PUPable_decl_template(              // NOLINT
      DummyBufferUpdater<AnalyticSolution>);  // NOLINT

  explicit DummyBufferUpdater(CkMigrateMessage* /*unused*/) noexcept
      : extraction_radius_{1.0},
        coordinate_amplitude_{0.0},
        coordinate_frequency_{0.0},
        l_max_{0} {}

  double update_buffers_for_time(
      const gsl::not_null<Variables<detail::cce_input_tags>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t interpolator_length, const size_t buffer_depth) const
      noexcept override {
    if (*time_span_end > interpolator_length and
        time_buffer_[*time_span_end - interpolator_length + 1] > time) {
      // the next time an update will be required
      return time_buffer_[*time_span_end - interpolator_length + 1];
    }
    // find the time spans that are needed
    auto new_span_pair = detail::create_span_for_time_value(
        time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
        time_buffer_);
    *time_span_start = new_span_pair.first;
    *time_span_end = new_span_pair.second;

    const size_t goldberg_size = square(l_max_ + 1);
    tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{goldberg_size};
    tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{
        goldberg_size};
    tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{
        goldberg_size};
    tnsr::I<ComplexModalVector, 3> shift_coefficients{goldberg_size};
    tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{goldberg_size};
    tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{goldberg_size};
    Scalar<ComplexModalVector> lapse_coefficients{goldberg_size};
    Scalar<ComplexModalVector> dt_lapse_coefficients{goldberg_size};
    Scalar<ComplexModalVector> dr_lapse_coefficients{goldberg_size};
    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution_, extraction_radius_,
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max_);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::SpatialMetric>(*buffers)),
          spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(*buffers)),
          dr_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<::Tags::dt<Tags::detail::SpatialMetric>>(*buffers)),
          dt_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Shift>(*buffers)),
          shift_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Dr<Tags::detail::Shift>>(*buffers)),
          dr_shift_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Shift>>(*buffers)),
          dt_shift_coefficients, time_index, *time_span_end - *time_span_start);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Lapse>(*buffers)),
          lapse_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Dr<Tags::detail::Lapse>>(*buffers)),
          dr_lapse_coefficients, time_index, *time_span_end - *time_span_start);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Lapse>>(*buffers)),
          dt_lapse_coefficients, time_index, *time_span_end - *time_span_start);
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }

  std::unique_ptr<WorldtubeBufferUpdater> get_clone() const noexcept override {
    return std::make_unique<DummyBufferUpdater>(*this);
  }

  bool time_is_outside_range(const double time) const noexcept override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const noexcept override { return l_max_; }

  double get_extraction_radius() const noexcept override {
    return extraction_radius_;
  }

  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

 private:
  template <typename... Structure>
  void update_tensor_buffer_with_tensor_at_time_index(
      const gsl::not_null<Tensor<ComplexModalVector, Structure...>*>
          tensor_buffer,
      const Tensor<ComplexModalVector, Structure...>& tensor_at_time,
      const size_t time_index, const size_t time_span_extent) const noexcept {
    for (size_t i = 0; i < tensor_at_time.size(); ++i) {
      for (size_t k = 0; k < tensor_at_time[i].size(); ++k) {
        (*tensor_buffer)[i][time_index + k * time_span_extent] =
            tensor_at_time[i][k];
      }
    }
  }

  DataVector time_buffer_;
  AnalyticSolution solution_;
  double extraction_radius_;
  double coordinate_amplitude_;
  double coordinate_frequency_;
  size_t l_max_;
};

template <>
PUP::able::PUP_ID
    Cce::DummyBufferUpdater<gr::Solutions::KerrSchild>::my_PUP_ID = 0;

template <typename Generator>
void test_data_manager_with_dummy_buffer_updater(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 8;
  const size_t interpolator_length = 3;
  const size_t l_max = 8;

  DataVector time_buffer{30};
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    time_buffer[i] = target_time - 1.55 + 0.1 * i;
  }

  WorldtubeDataManager boundary_data_manager{
      std::make_unique<DummyBufferUpdater<gr::Solutions::KerrSchild>>(
          time_buffer, solution, extraction_radius, amplitude, frequency,
          l_max),
      l_max, buffer_size,
      std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u)};
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // populate the test box using the boundary data manager that performs
  using boundary_variables_tag =
      ::Tags::Variables<Tags::characteristic_worldtube_boundary_tags>;

  auto expected_boundary_box =
      db::create<db::AddSimpleTags<boundary_variables_tag>>(
          db::item_type<boundary_variables_tag>{number_of_angular_points});
  auto interpolated_boundary_box =
      db::create<db::AddSimpleTags<boundary_variables_tag>>(
          db::item_type<boundary_variables_tag>{number_of_angular_points});

  boundary_data_manager.populate_hypersurface_boundary_data(
      make_not_null(&interpolated_boundary_box), target_time);

  // populate the expected box with the result from the analytic modes passed to
  // the boundary data computation.
  const size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};
  create_fake_time_varying_modal_data(
      make_not_null(&spatial_metric_coefficients),
      make_not_null(&dt_spatial_metric_coefficients),
      make_not_null(&dr_spatial_metric_coefficients),
      make_not_null(&shift_coefficients), make_not_null(&dt_shift_coefficients),
      make_not_null(&dr_shift_coefficients), make_not_null(&lapse_coefficients),
      make_not_null(&dt_lapse_coefficients),
      make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
      amplitude, frequency, target_time, l_max, false);

  create_bondi_boundary_data(
      make_not_null(&expected_boundary_box), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius, l_max);
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<Tags::characteristic_worldtube_boundary_tags>(
      [&expected_boundary_box, &interpolated_boundary_box,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(tag::name());
        const auto& test_lhs = db::get<tag>(expected_boundary_box);
        const auto& test_rhs = db::get<tag>(interpolated_boundary_box);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     angular_derivative_approx);
      });
}

template <typename Generator>
void test_spec_worldtube_buffer_updater(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  const double extraction_radius = 100;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50 * value_dist(*gen);

  const size_t buffer_size = 8;
  const size_t interpolator_length = 3;
  const size_t l_max = 8;

  Variables<detail::cce_input_tags> coefficients_buffers_from_file{
      (buffer_size + 2 * interpolator_length) * square(l_max + 1)};
  Variables<detail::cce_input_tags> expected_coefficients_buffers{
      (buffer_size + 2 * interpolator_length) * square(l_max + 1)};
  size_t goldberg_size = square(l_max + 1);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{goldberg_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{goldberg_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{goldberg_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{goldberg_size};
  Scalar<ComplexModalVector> lapse_coefficients{goldberg_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{goldberg_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{goldberg_size};

  // write times to file for several steps before and after the target time
  const std::string filename = "test_CceR0100.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  // scoped to close the file
  {
    TestHelpers::WorldtubeModeRecorder recorder{filename, l_max};
    for (size_t t = 0; t < 30; ++t) {
      const double time = 0.1 * t + target_time - 1.5;
      create_fake_time_varying_modal_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
          amplitude, frequency, time, l_max);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/g", i, j), time,
              spatial_metric_coefficients.get(i, j), l_max);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/Drg", i, j), time,
              dr_spatial_metric_coefficients.get(i, j), l_max);
          recorder.append_worldtube_mode_data(
              detail::dataset_name_for_component("/Dtg", i, j), time,
              dt_spatial_metric_coefficients.get(i, j), l_max);
        }
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/Shift", i), time,
            shift_coefficients.get(i), l_max);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DrShift", i), time,
            dr_shift_coefficients.get(i), l_max);
        recorder.append_worldtube_mode_data(
            detail::dataset_name_for_component("/DtShift", i), time,
            dt_shift_coefficients.get(i), l_max);
      }
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/Lapse"), time,
          get(lapse_coefficients), l_max);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/DrLapse"), time,
          get(dr_lapse_coefficients), l_max);
      recorder.append_worldtube_mode_data(
          detail::dataset_name_for_component("/DtLapse"), time,
          get(dt_lapse_coefficients), l_max);
    }
  }
  // request an appropriate buffer
  SpecWorldtubeH5BufferUpdater buffer_updater{filename};
  size_t time_span_start = 0;
  size_t time_span_end = 0;
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, interpolator_length, buffer_size);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  time_span_start = 0;
  time_span_end = 0;
  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 1.5 + 0.1 * i));
  }

  const DummyBufferUpdater<gr::Solutions::KerrSchild> dummy_buffer_updater{
      time_buffer, solution, extraction_radius, amplitude, frequency, l_max};
  dummy_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, interpolator_length, buffer_size);
  // check that the data in the buffer matches the expected analytic data.
  tmpl::for_each<detail::cce_input_tags>([
    &expected_coefficients_buffers, &coefficients_buffers_from_file
  ](auto tag_v) noexcept {
    using tag = typename decltype(tag_v)::type;
    INFO(tag::name());
    const auto& test_lhs = get<tag>(expected_coefficients_buffers);
    const auto& test_rhs = get<tag>(coefficients_buffers_from_file);
    CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
  });
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ReadBoundaryDataH5",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  test_spec_worldtube_buffer_updater(make_not_null(&gen));
  test_data_manager_with_dummy_buffer_updater(make_not_null(&gen));
}
}  // namespace Cce
