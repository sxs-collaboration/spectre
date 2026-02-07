// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "Helpers/Evolution/Systems/Cce/WriteToWorldtubeH5.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

template <typename T>
class DummyBufferUpdater  // NOLINT
    : public WorldtubeBufferUpdater<cce_metric_input_tags<T>> {
 public:
  DummyBufferUpdater()
      : extraction_radius_{1.0},
        coordinate_amplitude_{0.0},
        coordinate_frequency_{0.0},
        l_max_{0} {}
  DummyBufferUpdater(DataVector time_buffer,
                     const gr::Solutions::KerrSchild& solution,
                     const std::optional<double> extraction_radius,
                     const double coordinate_amplitude,
                     const double coordinate_frequency, const size_t l_max,
                     const bool apply_normalization_bug = false,
                     const bool has_version_history = true)
      : time_buffer_{std::move(time_buffer)},
        solution_{solution},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max},
        apply_normalization_bug_{apply_normalization_bug},
        has_version_history_{has_version_history} {}

  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(
      WorldtubeBufferUpdater<cce_metric_input_tags<T>>, DummyBufferUpdater);

  double update_buffers_for_time(
      const gsl::not_null<Variables<cce_metric_input_tags<T>>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t /*l_max*/, const size_t interpolator_length,
      const size_t buffer_depth,
      const bool time_varies_fastest = true) const override {
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

    const size_t size =
        std::is_same_v<T, ComplexModalVector>
            ? square(l_max_ + 1)
            : Spectral::Swsh::number_of_swsh_collocation_points(l_max_);
    tnsr::ii<T, 3> spatial_metric_coefficients{size};
    tnsr::ii<T, 3> dt_spatial_metric_coefficients{size};
    tnsr::ii<T, 3> dr_spatial_metric_coefficients{size};
    tnsr::I<T, 3> shift_coefficients{size};
    tnsr::I<T, 3> dt_shift_coefficients{size};
    tnsr::I<T, 3> dr_shift_coefficients{size};
    Scalar<T> lapse_coefficients{size};
    Scalar<T> dt_lapse_coefficients{size};
    Scalar<T> dr_lapse_coefficients{size};
    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      TestHelpers::create_fake_time_varying_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution_,
          extraction_radius_.value_or(default_extraction_radius_),
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max_, true,
          apply_normalization_bug_);

      ASSERT(get(lapse_coefficients).size() == size,
             "Oh no... Expected = " << size << ", got "
                                    << get(lapse_coefficients).size());

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::SpatialMetric<T>>(*buffers)),
          spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start, time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<Tags::detail::Dr<Tags::detail::SpatialMetric<T>>>(*buffers)),
          dr_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start, time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<::Tags::dt<Tags::detail::SpatialMetric<T>>>(*buffers)),
          dt_spatial_metric_coefficients, time_index,
          *time_span_end - *time_span_start, time_varies_fastest);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Shift<T>>(*buffers)),
          shift_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<Tags::detail::Dr<Tags::detail::Shift<T>>>(*buffers)),
          dr_shift_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Shift<T>>>(*buffers)),
          dt_shift_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);

      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<Tags::detail::Lapse<T>>(*buffers)),
          lapse_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(
              &get<Tags::detail::Dr<Tags::detail::Lapse<T>>>(*buffers)),
          dr_lapse_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);
      update_tensor_buffer_with_tensor_at_time_index(
          make_not_null(&get<::Tags::dt<Tags::detail::Lapse<T>>>(*buffers)),
          dt_lapse_coefficients, time_index, *time_span_end - *time_span_start,
          time_varies_fastest);
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }

  std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags<T>>> get_clone()
      const override {
    return std::make_unique<DummyBufferUpdater>(*this);
  }

  bool time_is_outside_range(const double time) const override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const override { return l_max_; }

  double get_extraction_radius() const override {
    return extraction_radius_.value_or(default_extraction_radius_);
  }

  bool has_version_history() const override { return has_version_history_; }

  DataVector& get_time_buffer() override { return time_buffer_; }

  void pup(PUP::er& p) override {
    p | time_buffer_;
    p | solution_;
    p | extraction_radius_;
    p | coordinate_amplitude_;
    p | coordinate_frequency_;
    p | default_extraction_radius_;
    p | l_max_;
    p | apply_normalization_bug_;
    p | has_version_history_;
  }

 private:
  template <typename... Structure>
  void update_tensor_buffer_with_tensor_at_time_index(
      const gsl::not_null<Tensor<Structure...>*> tensor_buffer,
      const Tensor<Structure...>& tensor_at_time, const size_t time_index,
      const size_t time_span_extent, const bool time_varies_fastest) const {
    for (size_t i = 0; i < tensor_at_time.size(); ++i) {
      for (size_t k = 0; k < tensor_at_time[i].size(); ++k) {
        const size_t buffer_index =
            time_varies_fastest ? time_index + k * time_span_extent
                                : k + time_index * tensor_at_time[i].size();
        (*tensor_buffer)[i][buffer_index] = tensor_at_time[i][k];
      }
    }
  }

  DataVector time_buffer_;
  gr::Solutions::KerrSchild solution_;
  std::optional<double> extraction_radius_;
  double default_extraction_radius_ = 100.0;
  double coordinate_amplitude_;
  double coordinate_frequency_;
  size_t l_max_;
  bool apply_normalization_bug_ = false;
  bool has_version_history_ = true;
};

template <typename T>
class BondiBufferUpdater
    : public WorldtubeBufferUpdater<tmpl::conditional_t<
          std::is_same_v<T, ComplexModalVector>,
          Tags::worldtube_boundary_tags_for_writing<
              Spectral::Swsh::Tags::SwshTransform>,
          Tags::worldtube_boundary_tags_for_writing<Tags::BoundaryValue>>> {
 public:
  using tags_for_writing = tmpl::conditional_t<
      std::is_same_v<T, ComplexModalVector>,
      Tags::worldtube_boundary_tags_for_writing<
          Spectral::Swsh::Tags::SwshTransform>,
      Tags::worldtube_boundary_tags_for_writing<Tags::BoundaryValue>>;

  BondiBufferUpdater() = default;
  BondiBufferUpdater(DataVector time_buffer,
                     const gr::Solutions::KerrSchild& solution,
                     const std::optional<double> extraction_radius,
                     const double coordinate_amplitude,
                     const double coordinate_frequency, const size_t l_max,
                     const bool /*unused*/ = false)
      : time_buffer_{std::move(time_buffer)},
        solution_{solution},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max} {}

  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(WorldtubeBufferUpdater<tags_for_writing>,
                                     BondiBufferUpdater);

  double update_buffers_for_time(
      const gsl::not_null<Variables<tags_for_writing>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t l_max, const size_t interpolator_length,
      const size_t buffer_depth,
      const bool time_varies_fastest = true) const override {
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

    const size_t libsharp_size =
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max);
    tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
    tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{
        libsharp_size};
    tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{
        libsharp_size};
    tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
    tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
    tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
    Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
    Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
    Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};

    const size_t number_of_angular_points =
        Spectral::Swsh::number_of_swsh_collocation_points(l_max);
    Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
        boundary_variables{number_of_angular_points};

    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      TestHelpers::create_fake_time_varying_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution_,
          extraction_radius_.value_or(default_extraction_radius_),
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max_, false);

      Cce::create_bondi_boundary_data(
          make_not_null(&boundary_variables), spatial_metric_coefficients,
          dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
          shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
          lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
          extraction_radius_.value_or(default_extraction_radius_), l_max);
      tmpl::for_each<tmpl::transform<
          tags_for_writing, tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
          [&, this](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            SpinWeighted<T, tag::type::type::spin> spin_weighted_at_time{};
            if constexpr (std::is_same_v<T, ComplexModalVector>) {
              spin_weighted_at_time =
                  Spectral::Swsh::libsharp_to_goldberg_modes(
                      Spectral::Swsh::swsh_transform(
                          l_max, 1,
                          get(get<Tags::BoundaryValue<tag>>(
                              boundary_variables))),
                      l_max);
            } else {
              (void)l_max;
              spin_weighted_at_time =
                  get(get<Tags::BoundaryValue<tag>>(boundary_variables));
            }
            this->update_buffer_with_scalar_at_time_index(
                make_not_null(&get<tmpl::conditional_t<
                                  std::is_same_v<T, ComplexModalVector>,
                                  Spectral::Swsh::Tags::SwshTransform<tag>,
                                  Tags::BoundaryValue<tag>>>(*buffers)),
                spin_weighted_at_time, time_index,
                *time_span_end - *time_span_start, time_varies_fastest);
          });
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }
  std::unique_ptr<WorldtubeBufferUpdater<tags_for_writing>> get_clone()
      const override {
    return std::make_unique<BondiBufferUpdater>(*this);
  }

  bool time_is_outside_range(const double time) const override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  size_t get_l_max() const override { return l_max_; }

  double get_extraction_radius() const override {
    return extraction_radius_.value_or(default_extraction_radius_);
  }

  DataVector& get_time_buffer() override { return time_buffer_; }

  bool has_version_history() const override { return true; }

  void pup(PUP::er& p) override {
    p | time_buffer_;
    p | solution_;
    p | extraction_radius_;
    p | coordinate_amplitude_;
    p | coordinate_frequency_;
    p | l_max_;
  }

 private:
  template <int Spin>
  void update_buffer_with_scalar_at_time_index(
      const gsl::not_null<Scalar<SpinWeighted<T, Spin>>*> scalar_buffer,
      const SpinWeighted<T, Spin>& spin_weighted_at_time,
      const size_t time_index, const size_t time_span_extent,
      const bool time_varies_fastest) const {
    for (size_t k = 0; k < spin_weighted_at_time.size(); ++k) {
      const size_t buffer_index =
          time_varies_fastest ? time_index + k * time_span_extent
                              : k + time_index * spin_weighted_at_time.size();
      get(*scalar_buffer).data()[buffer_index] =
          spin_weighted_at_time.data()[k];
    }
  }

  DataVector time_buffer_;
  gr::Solutions::KerrSchild solution_;
  std::optional<double> extraction_radius_;
  double default_extraction_radius_ = 100.0;
  double coordinate_amplitude_ = 0.0;
  double coordinate_frequency_ = 0.0;
  size_t l_max_ = 0;
};

#if defined(SPECTRE_USE_CHARM)
template <typename T>
PUP::able::PUP_ID Cce::DummyBufferUpdater<T>::my_PUP_ID = 0;  // NOLINT
template <typename T>
PUP::able::PUP_ID Cce::BondiBufferUpdater<T>::my_PUP_ID = 0;  // NOLINT
#endif  // SPECTRE_USE_CHARM

template class Cce::DummyBufferUpdater<ComplexModalVector>;
template class Cce::DummyBufferUpdater<DataVector>;
template class Cce::BondiBufferUpdater<ComplexModalVector>;

namespace {

template <typename DataManager, typename DummyUpdater, typename Generator>
void test_data_manager_with_bondi_buffer_updater(
    const gsl::not_null<Generator*> gen,
    const bool apply_normalization_bug = false, const bool descending_m = true,
    const std::optional<double> extraction_radius = std::nullopt) {
  // note that the default_extraction_radius is what will be reported
  // from the buffer updater when the extraction_radius is the default
  // `std::nullopt`.
  const double default_extraction_radius = 100.0;
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t l_max = 8;

  DataVector time_buffer{30};
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    time_buffer[i] = target_time - 1.55 + 0.1 * static_cast<double>(i);
  }

  DataManager boundary_data_manager;
  if constexpr (std::is_same_v<DataManager, MetricWorldtubeDataManager>) {
    if (not apply_normalization_bug) {
      boundary_data_manager = DataManager{
          std::make_unique<DummyUpdater>(time_buffer, solution,
                                         extraction_radius, amplitude,
                                         frequency, l_max, false, descending_m),
          l_max, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          descending_m};
    } else {
      boundary_data_manager = DataManager{
          std::make_unique<DummyUpdater>(time_buffer, solution,
                                         extraction_radius, amplitude,
                                         frequency, l_max, true, false),
          l_max, buffer_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u),
          descending_m};
    }
  } else {
    // avoid compiler warnings in the case where the normalization bug booleans
    // aren't used.
    (void)apply_normalization_bug;
    (void)descending_m;
    boundary_data_manager = DataManager{
        std::make_unique<DummyUpdater>(time_buffer, solution, extraction_radius,
                                       amplitude, frequency, l_max, false),
        l_max, buffer_size,
        std::make_unique<intrp::BarycentricRationalSpanInterpolator>(3u, 4u)};
  }
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      expected_boundary_variables{number_of_angular_points};
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      interpolated_boundary_variables{number_of_angular_points};

  Parallel::NodeLock hdf5_lock{};
  boundary_data_manager.populate_hypersurface_boundary_data(
      make_not_null(&interpolated_boundary_variables), target_time,
      make_not_null(&hdf5_lock));

  // populate the expected variables with the result from the analytic modes
  // passed to the boundary data computation.
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
  TestHelpers::create_fake_time_varying_data(
      make_not_null(&spatial_metric_coefficients),
      make_not_null(&dt_spatial_metric_coefficients),
      make_not_null(&dr_spatial_metric_coefficients),
      make_not_null(&shift_coefficients), make_not_null(&dt_shift_coefficients),
      make_not_null(&dr_shift_coefficients), make_not_null(&lapse_coefficients),
      make_not_null(&dt_lapse_coefficients),
      make_not_null(&dr_lapse_coefficients), solution,
      extraction_radius.value_or(default_extraction_radius), amplitude,
      frequency, target_time, l_max, false);

  create_bondi_boundary_data(
      make_not_null(&expected_boundary_variables), spatial_metric_coefficients,
      dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
      shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
      lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
      extraction_radius.value_or(default_extraction_radius), l_max);
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  tmpl::for_each<
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      [&expected_boundary_variables, &interpolated_boundary_variables,
       &angular_derivative_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_boundary_variables);
        const auto& test_rhs = get<tag>(interpolated_boundary_variables);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs,
                                     angular_derivative_approx);
      });
}

template <typename T, typename Generator>
void test_metric_worldtube_buffer_updater_impl(
    const gsl::not_null<Generator*> gen,
    const bool extraction_radius_in_filename, const bool time_varies_fastest,
    const std::optional<bool>& extra_adm = std::nullopt) {
  constexpr bool is_modal = std::is_same_v<T, ComplexModalVector>;
  CAPTURE(is_modal);
  CAPTURE(extraction_radius_in_filename);
  CAPTURE(time_varies_fastest);
  CAPTURE(extra_adm);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};
  CAPTURE(mass);
  CAPTURE(spin);
  CAPTURE(center);

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  // If using adm options, need amplitude to be zero since this will change the
  // coordinates that the variables are at, and we assume a constant radius
  const double amplitude = extra_adm.has_value() ? 0.0 : 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  CAPTURE(frequency);
  CAPTURE(amplitude);
  CAPTURE(target_time);

  const size_t buffer_size = 4;
  const size_t interpolator_length = 2;
  const size_t file_l_max = 8;
  // Must be the same as file_l_max for nodal data
  const size_t computation_l_max = is_modal ? file_l_max + 2 : file_l_max;
  CAPTURE(buffer_size);
  CAPTURE(interpolator_length);
  CAPTURE(file_l_max);
  CAPTURE(computation_l_max);

  const size_t computation_buffer_size =
      (buffer_size + 2 * interpolator_length) *
      (is_modal ? square(computation_l_max + 1)
                : Spectral::Swsh::number_of_swsh_collocation_points(
                      computation_l_max));
  Variables<cce_metric_input_tags<T>> coefficients_buffers_from_file{
      computation_buffer_size};
  Variables<cce_metric_input_tags<T>> expected_coefficients_buffers{
      computation_buffer_size};
  const std::string filename = extraction_radius_in_filename
                                   ? "BoundaryDataH5Test_CceR0100.h5"
                                   : "BoundaryDataH5Test.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  TestHelpers::write_test_file<T, false>(
      solution, filename, target_time, extraction_radius, frequency, amplitude,
      file_l_max, true, extra_adm.has_value());

  using AdmOptions = typename MetricWorldtubeH5BufferUpdater<T>::AdmOptions;
  std::optional<AdmOptions> adm_options{};

  if (extra_adm.has_value()) {
    using Lapse = typename AdmOptions::Lapse;
    using Shift = typename AdmOptions::Shift;
    const Lapse lapse{true};
    if (extra_adm.value()) {
      const Shift shift{true, 0.75};
      adm_options = AdmOptions{lapse, shift};
    } else {
      const Shift shift{true, 2.0, 1.0};
      adm_options = AdmOptions{lapse, shift};
    }
  }
  // request an appropriate buffer
  auto buffer_updater =
      extraction_radius_in_filename
          ? MetricWorldtubeH5BufferUpdater<T>{filename, std::nullopt, true,
                                              adm_options}
          : MetricWorldtubeH5BufferUpdater<T>{filename, extraction_radius, true,
                                              adm_options};
  auto serialized_and_deserialized_updater =
      serialize_and_deserialize(buffer_updater);
  size_t time_span_start = 0;
  size_t time_span_end = 0;
  if (not is_modal) {
    CHECK_THROWS_WITH(
        buffer_updater.update_buffers_for_time(
            make_not_null(&coefficients_buffers_from_file),
            make_not_null(&time_span_start), make_not_null(&time_span_end),
            target_time, 2 * computation_l_max, interpolator_length,
            buffer_size, time_varies_fastest),
        Catch::Matchers::ContainsSubstring(
            "When reading in nodal data, the LMax that "
            "MetricWorldtubeH5BufferUpdater was constructed with"));
    time_span_start = 0;
    time_span_end = 0;
    if (extra_adm.has_value()) {
      CHECK_THROWS_WITH(
          buffer_updater.update_buffers_for_time(
              make_not_null(&coefficients_buffers_from_file),
              make_not_null(&time_span_start), make_not_null(&time_span_end),
              target_time, computation_l_max, interpolator_length, buffer_size,
              true),
          Catch::Matchers::ContainsSubstring("Time must not vary fastest"));
      time_span_start = 0;
      time_span_end = 0;
    }
  }
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size,
      time_varies_fastest);

  Variables<cce_metric_input_tags<T>> coefficients_buffers_from_serialized{
      computation_buffer_size};
  size_t time_span_start_from_serialized = 0;
  size_t time_span_end_from_serialized = 0;
  serialized_and_deserialized_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_serialized),
      make_not_null(&time_span_start_from_serialized),
      make_not_null(&time_span_end_from_serialized), target_time,
      computation_l_max, interpolator_length, buffer_size, time_varies_fastest);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  time_span_start = 0;
  time_span_end = 0;
  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 1.5 + 0.1 * i));
  }
  const auto& time_buffer_from_serialized =
      serialized_and_deserialized_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer_from_serialized[i] ==
          approx(target_time - 1.5 + 0.1 * i));
  }

  const DummyBufferUpdater<T> bondi_buffer_updater = serialize_and_deserialize(
      DummyBufferUpdater<T>{time_buffer, solution, extraction_radius, amplitude,
                            frequency, computation_l_max});
  bondi_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size,
      time_varies_fastest);

  // check that the data in the buffer matches the expected analytic data.
  tmpl::for_each<cce_metric_input_tags<T>>(
      [&expected_coefficients_buffers, &coefficients_buffers_from_file,
       &coefficients_buffers_from_serialized, &extra_adm](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        // Since we don't have an expected value for the time derivatives of the
        // lapse or shift because of the 1+log slicing and gamma-driver
        // conditions, we just don't check them here. All other values should be
        // the same though
        if (extra_adm.has_value() and
            (std::is_same_v<tag, ::Tags::dt<Tags::detail::Shift<T>>> or
             std::is_same_v<tag, ::Tags::dt<Tags::detail::Lapse<T>>>)) {
          return;
        }
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_coefficients_buffers);
        const auto& test_rhs = get<tag>(coefficients_buffers_from_file);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs);
        const auto& test_rhs_from_serialized =
            get<tag>(coefficients_buffers_from_serialized);
        CHECK_ITERABLE_APPROX(test_lhs, test_rhs_from_serialized);
      });
  CHECK(buffer_updater.get_extraction_radius() == 100.0);
}

template <typename T, typename Generator>
void test_bondi_worldtube_buffer_updater_impl(
    const gsl::not_null<Generator*> gen,
    const bool extraction_radius_in_filename, const bool time_varies_fastest) {
  constexpr bool is_modal = std::is_same_v<T, ComplexModalVector>;
  CAPTURE(is_modal);
  CAPTURE(extraction_radius_in_filename);
  CAPTURE(time_varies_fastest);
  UniformCustomDistribution<double> value_dist{0.1, 0.5};
  // first prepare the input for the modal version
  const double mass = value_dist(*gen);
  const std::array<double, 3> spin{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  const std::array<double, 3> center{
      {value_dist(*gen), value_dist(*gen), value_dist(*gen)}};
  gr::Solutions::KerrSchild solution{mass, spin, center};
  CAPTURE(mass);
  CAPTURE(spin);
  CAPTURE(center);

  const double extraction_radius = 100.0;

  // acceptable parameters for the fake sinusoid variation in the input
  // parameters
  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);
  CAPTURE(frequency);
  CAPTURE(amplitude);
  CAPTURE(target_time);

  const size_t buffer_size = 4;
  const size_t interpolator_length = 3;
  const size_t file_l_max = 8;
  const size_t computation_l_max = is_modal ? file_l_max + 2 : file_l_max;
  CAPTURE(buffer_size);
  CAPTURE(interpolator_length);
  CAPTURE(file_l_max);
  CAPTURE(computation_l_max);

  using tags_for_writing =
      typename BondiWorldtubeH5BufferUpdater<T>::tags_for_writing;

  const size_t computation_buffer_size =
      (buffer_size + 2 * interpolator_length) *
      (is_modal ? square(computation_l_max + 1)
                : Spectral::Swsh::number_of_swsh_collocation_points(
                      computation_l_max));
  Variables<tags_for_writing> coefficients_buffers_from_file{
      computation_buffer_size};
  Variables<tags_for_writing> expected_coefficients_buffers{
      computation_buffer_size};
  size_t libsharp_size =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(file_l_max);
  tnsr::ii<ComplexModalVector, 3> spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dt_spatial_metric_coefficients{libsharp_size};
  tnsr::ii<ComplexModalVector, 3> dr_spatial_metric_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dt_shift_coefficients{libsharp_size};
  tnsr::I<ComplexModalVector, 3> dr_shift_coefficients{libsharp_size};
  Scalar<ComplexModalVector> lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dt_lapse_coefficients{libsharp_size};
  Scalar<ComplexModalVector> dr_lapse_coefficients{libsharp_size};

  const size_t file_number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(file_l_max);
  Variables<Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>
      boundary_data_variables{file_number_of_angular_points};

  // write times to file for several steps before and after the target time
  const std::string filename = extraction_radius_in_filename
                                   ? "BoundaryDataH5Test_CceR0100.h5"
                                   : "BoundaryDataH5Test.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  // scoped to close the file
  {
    using RecorderType =
        tmpl::conditional_t<is_modal, Cce::WorldtubeModeRecorder,
                            Cce::TestHelpers::WorldtubeModeRecorder>;
    RecorderType recorder{file_l_max, filename};
    for (size_t t = 0; t < 20; ++t) {
      const double time = 0.01 * static_cast<double>(t) + target_time - 0.1;
      TestHelpers::create_fake_time_varying_data(
          make_not_null(&spatial_metric_coefficients),
          make_not_null(&dt_spatial_metric_coefficients),
          make_not_null(&dr_spatial_metric_coefficients),
          make_not_null(&shift_coefficients),
          make_not_null(&dt_shift_coefficients),
          make_not_null(&dr_shift_coefficients),
          make_not_null(&lapse_coefficients),
          make_not_null(&dt_lapse_coefficients),
          make_not_null(&dr_lapse_coefficients), solution, extraction_radius,
          amplitude, frequency, time, file_l_max, false);

      create_bondi_boundary_data(
          make_not_null(&boundary_data_variables), spatial_metric_coefficients,
          dt_spatial_metric_coefficients, dr_spatial_metric_coefficients,
          shift_coefficients, dt_shift_coefficients, dr_shift_coefficients,
          lapse_coefficients, dt_lapse_coefficients, dr_lapse_coefficients,
          extraction_radius, file_l_max);

      // loop over the tags that we want to dump.
      tmpl::for_each<Cce::Tags::worldtube_boundary_tags_for_writing<>>(
          [&recorder, &boundary_data_variables, &time](auto tag_v) {
            using tag = typename decltype(tag_v)::type;

            const ComplexDataVector& nodal_data =
                get(get<tag>(boundary_data_variables)).data();

            if constexpr (is_modal) {
              recorder.template append_modal_data<tag::type::type::spin>(
                  dataset_label_for_tag<typename tag::tag>(), time, nodal_data,
                  file_l_max);
            } else {
              //   // This will write nodal data
              recorder.append_worldtube_mode_data(
                  dataset_label_for_tag<typename tag::tag>(), time, nodal_data);
            }
          });
    }
  }
  // request an appropriate buffer
  auto buffer_updater =
      extraction_radius_in_filename
          ? BondiWorldtubeH5BufferUpdater<T>{filename}
          : BondiWorldtubeH5BufferUpdater<T>{filename, extraction_radius};
  CHECK(buffer_updater.get_l_max() == file_l_max);
  auto serialized_and_deserialized_updater =
      serialize_and_deserialize(buffer_updater);
  size_t time_span_start = 0;
  size_t time_span_end = 0;
  if (not is_modal) {
    CHECK_THROWS_WITH(
        buffer_updater.update_buffers_for_time(
            make_not_null(&coefficients_buffers_from_file),
            make_not_null(&time_span_start), make_not_null(&time_span_end),
            target_time, 2 * computation_l_max, interpolator_length,
            buffer_size, time_varies_fastest),
        Catch::Matchers::ContainsSubstring(
            "When reading in nodal data, the LMax that the BufferUpdater was "
            "constructed with"));
    time_span_start = 0;
    time_span_end = 0;
  }
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size,
      time_varies_fastest);

  Variables<tags_for_writing> coefficients_buffers_from_serialized{
      computation_buffer_size};
  size_t time_span_start_from_serialized = 0;
  size_t time_span_end_from_serialized = 0;
  serialized_and_deserialized_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_serialized),
      make_not_null(&time_span_start_from_serialized),
      make_not_null(&time_span_end_from_serialized), target_time,
      computation_l_max, interpolator_length, buffer_size, time_varies_fastest);

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
  time_span_start = 0;
  time_span_end = 0;
  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 0.1 + 0.01 * i));
  }
  const auto& time_buffer_from_serialized =
      serialized_and_deserialized_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer_from_serialized[i] ==
          approx(target_time - 0.1 + 0.01 * i));
  }

  const BondiBufferUpdater<T> bondi_buffer_updater = serialize_and_deserialize(
      BondiBufferUpdater<T>{time_buffer, solution, extraction_radius, amplitude,
                            frequency, computation_l_max});
  bondi_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size,
      time_varies_fastest);

  // this approximation needs to be comparatively loose because it is comparing
  // modes, which tend to have the error set by the scale of the original
  // collocation errors (so, the dominant modes), rather than the scale of the
  // individual mode being examined.
  Approx modal_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e5)
          .scale(1.0);

  // check that the data in the buffer matches the expected analytic data.
  tmpl::for_each<tags_for_writing>(
      [&expected_coefficients_buffers, &coefficients_buffers_from_file,
       &coefficients_buffers_from_serialized, &modal_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& test_lhs = get<tag>(expected_coefficients_buffers);
        const auto& test_rhs = get<tag>(coefficients_buffers_from_file);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs, modal_approx);
        const auto& test_rhs_from_serialized =
            get<tag>(coefficients_buffers_from_serialized);
        CHECK_ITERABLE_CUSTOM_APPROX(test_lhs, test_rhs_from_serialized,
                                     modal_approx);
      });
  CHECK(buffer_updater.get_extraction_radius() == 100.0);
}

template <typename Generator>
void test_metric_worldtube_buffer_updater(const gsl::not_null<Generator*> gen) {
  INFO("SpEC worldtube (aka metric)");
  for (const auto& [extraction_radius_in_filename, time_varies_fastest] :
       cartesian_product(std::array{true, false}, std::array{true, false})) {
    test_metric_worldtube_buffer_updater_impl<ComplexModalVector>(
        gen, extraction_radius_in_filename, time_varies_fastest);
    test_metric_worldtube_buffer_updater_impl<DataVector>(
        gen, extraction_radius_in_filename, time_varies_fastest);
  }

  {
    INFO("First order form");
    test_metric_worldtube_buffer_updater_impl<DataVector>(gen, true, false,
                                                          {true});
  }
  {
    INFO("Conformal Christoffel form");
    test_metric_worldtube_buffer_updater_impl<DataVector>(gen, true, false,
                                                          {false});
  }
}

template <typename Generator>
void test_bondi_worldtube_buffer_updater(const gsl::not_null<Generator*> gen) {
  INFO("Reduced SpEC worldtube (aka Bondi)");
  for (const auto& [extraction_radius_in_filename, time_varies_fastest] :
       cartesian_product(std::array{true, false}, std::array{true, false})) {
    test_bondi_worldtube_buffer_updater_impl<ComplexModalVector>(
        gen, extraction_radius_in_filename, time_varies_fastest);
    test_bondi_worldtube_buffer_updater_impl<ComplexDataVector>(
        gen, extraction_radius_in_filename, time_varies_fastest);
  }
}
}  // namespace

// An increased timeout because this test seems to have high variance in
// duration. The high variance may be due to the comparatively high magnitude of
// disk operations in this test.
// [[TimeOut, 60]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ReadBoundaryDataH5",
                  "[Unit][Cce]") {
  register_derived_classes_with_charm<
      Cce::WorldtubeBufferUpdater<cce_metric_input_tags<ComplexModalVector>>>();
  register_derived_classes_with_charm<
      Cce::WorldtubeBufferUpdater<Tags::worldtube_boundary_tags_for_writing<
          Spectral::Swsh::Tags::SwshTransform>>>();
  register_derived_classes_with_charm<Cce::WorldtubeDataManager<
      Cce::Tags::characteristic_worldtube_boundary_tags<
          Cce::Tags::BoundaryValue>>>();
  register_derived_classes_with_charm<intrp::SpanInterpolator>();
  MAKE_GENERATOR(gen);
  {
    INFO("Testing buffer updaters");
    test_metric_worldtube_buffer_updater(make_not_null(&gen));
    test_bondi_worldtube_buffer_updater(make_not_null(&gen));
  }
  {
    INFO("Testing data managers");
    using DummyBufferUpdater = DummyBufferUpdater<ComplexModalVector>;
    using BondiBufferUpdater = BondiBufferUpdater<ComplexModalVector>;
    test_data_manager_with_bondi_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen));
    // with normalization bug applied:
    test_data_manager_with_bondi_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), true, true);
    test_data_manager_with_bondi_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, true);
    test_data_manager_with_bondi_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, false);
    // check the case for an explicitly provided extraction radius.
    test_data_manager_with_bondi_buffer_updater<MetricWorldtubeDataManager,
                                                DummyBufferUpdater>(
        make_not_null(&gen), false, false, 200.0);
    test_data_manager_with_bondi_buffer_updater<BondiWorldtubeDataManager,
                                                BondiBufferUpdater>(
        make_not_null(&gen));
  }
}
}  // namespace Cce
