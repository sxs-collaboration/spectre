// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"
#include "Evolution/Systems/Cce/WorldtubeModeRecorder.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Cce/BoundaryTestHelpers.hpp"
#include "Helpers/Evolution/Systems/Cce/KleinGordonBoundaryTestHelpers.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace Cce {

// A dummy buffer updater that holds reference worldtube data to compare with
class KleinGordonDummyBufferUpdater
    : public WorldtubeBufferUpdater<klein_gordon_input_tags> {
 public:
  KleinGordonDummyBufferUpdater(DataVector time_buffer,
                                const std::optional<double> extraction_radius,
                                const double coordinate_amplitude,
                                const double coordinate_frequency,
                                const size_t l_max)
      : time_buffer_{std::move(time_buffer)},
        extraction_radius_{extraction_radius},
        coordinate_amplitude_{coordinate_amplitude},
        coordinate_frequency_{coordinate_frequency},
        l_max_{l_max} {}

  WRAPPED_PUPable_decl_template(KleinGordonDummyBufferUpdater);  // NOLINT

  explicit KleinGordonDummyBufferUpdater(CkMigrateMessage* /*unused*/) {}

  double update_buffers_for_time(
      const gsl::not_null<Variables<klein_gordon_input_tags>*> buffers,
      const gsl::not_null<size_t*> time_span_start,
      const gsl::not_null<size_t*> time_span_end, const double time,
      const size_t l_max, const size_t interpolator_length,
      const size_t buffer_depth) const override {
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

    Scalar<ComplexModalVector> kg_psi_modal;
    Scalar<ComplexModalVector> kg_pi_modal;
    Scalar<DataVector> kg_psi_nodal;
    Scalar<DataVector> kg_pi_nodal;

    for (size_t time_index = 0; time_index < *time_span_end - *time_span_start;
         ++time_index) {
      TestHelpers::create_fake_time_varying_klein_gordon_data(
          make_not_null(&kg_psi_modal), make_not_null(&kg_pi_modal),
          make_not_null(&kg_psi_nodal), make_not_null(&kg_pi_nodal),
          extraction_radius_.value_or(default_extraction_radius_),
          coordinate_amplitude_, coordinate_frequency_,
          time_buffer_[time_index + *time_span_start], l_max);

      this->update_buffer_with_scalar_at_time_index(
          make_not_null(&get<Spectral::Swsh::Tags::SwshTransform<
                            Cce::Tags::KleinGordonPsi>>(*buffers)),
          kg_psi_modal, time_index, *time_span_end - *time_span_start);

      this->update_buffer_with_scalar_at_time_index(
          make_not_null(&get<Spectral::Swsh::Tags::SwshTransform<
                            Cce::Tags::KleinGordonPi>>(*buffers)),
          kg_pi_modal, time_index, *time_span_end - *time_span_start);
    }
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }

  std::unique_ptr<WorldtubeBufferUpdater<klein_gordon_input_tags>> get_clone()
      const override {
    return std::make_unique<KleinGordonDummyBufferUpdater>(*this);
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
    p | extraction_radius_;
    p | coordinate_amplitude_;
    p | coordinate_frequency_;
    p | l_max_;
  }

 private:
  template <int Spin>
  void update_buffer_with_scalar_at_time_index(
      const gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, Spin>>*>
          scalar_buffer,
      const Scalar<ComplexModalVector>& spin_weighted_at_time,
      const size_t time_index, const size_t time_span_extent) const {
    for (size_t k = 0; k < get(spin_weighted_at_time).size(); ++k) {
      get(*scalar_buffer).data()[time_index + k * time_span_extent] =
          get(spin_weighted_at_time)[k];
    }
  }

  DataVector time_buffer_;
  std::optional<double> extraction_radius_;
  double default_extraction_radius_ = 100.0;
  double coordinate_amplitude_ = 0.0;
  double coordinate_frequency_ = 0.0;
  size_t l_max_ = 0;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID Cce::KleinGordonDummyBufferUpdater::my_PUP_ID = 0;

namespace {

// This function tests `KleinGordonWorldtubeDataManager`, with special focus on
// its interpolator. Its private member `buffer_updater_` is created from the
// dummy buffer updater `KleinGordonDummyBufferUpdater` defined above, instead
// of `KleinGordonWorldtubeH5BufferUpdater`.  This allows us to stick
// exclusively to the test of `KleinGordonWorldtubeDataManager`. The test of
// `KleinGordonWorldtubeH5BufferUpdater` is performed below in
// `test_klein_gordon_worldtube_buffer_updater`.
//
// The function first generates nodal and modal data for the scalar field `psi`
// and its time derivative `pi` for a range of time stamps; and then uses the
// interpolator of `KleinGordonWorldtubeDataManager` to interpolate the data to
// a different time (`target_time`). Finally, it compares the interpolated
// results with the expected ones.
template <typename Generator>
void test_klein_gordon_data_manager_with_dummy_buffer_updater(
    const gsl::not_null<Generator*> gen) {
  const double extraction_radius = 100.0;
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t l_max = 8;

  DataVector time_buffer{30};
  // `target_time` is not an element of `time_buffer`, so the interpolation is
  // non-trivial.
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    time_buffer[i] = target_time - 1.55 + 0.1 * static_cast<double>(i);
  }

  // use `KleinGordonWorldtubeDataManager` to interpolate data
  KleinGordonWorldtubeDataManager boundary_data_manager;

  boundary_data_manager = KleinGordonWorldtubeDataManager{
      std::make_unique<KleinGordonDummyBufferUpdater>(
          time_buffer, extraction_radius, amplitude, frequency, l_max),
      l_max, buffer_size,
      std::make_unique<intrp::BarycentricRationalSpanInterpolator>(8u, 10u)};

  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  Variables<Tags::klein_gordon_worldtube_boundary_tags>
      interpolated_boundary_variables{number_of_angular_points};

  Parallel::NodeLock hdf5_lock{};
  boundary_data_manager.populate_hypersurface_boundary_data(
      make_not_null(&interpolated_boundary_variables), target_time,
      make_not_null(&hdf5_lock));

  // populate the expected variables with the result from the analytic modes
  // passed to the boundary data computation.
  Scalar<ComplexModalVector> kg_psi_modal;
  Scalar<ComplexModalVector> kg_pi_modal;
  Scalar<DataVector> kg_psi_nodal;
  Scalar<DataVector> kg_pi_nodal;

  TestHelpers::create_fake_time_varying_klein_gordon_data(
      make_not_null(&kg_psi_modal), make_not_null(&kg_pi_modal),
      make_not_null(&kg_psi_nodal), make_not_null(&kg_pi_nodal),
      extraction_radius, amplitude, frequency, target_time, l_max);

  // comparison
  Approx angular_derivative_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e3)
          .scale(1.0);

  const auto& interpolated_psi =
      get<Cce::Tags::BoundaryValue<Cce::Tags::KleinGordonPsi>>(
          interpolated_boundary_variables);
  CHECK_ITERABLE_CUSTOM_APPROX(get(kg_psi_nodal), get(interpolated_psi).data(),
                               angular_derivative_approx);

  const auto& interpolated_pi =
      get<Cce::Tags::BoundaryValue<Cce::Tags::KleinGordonPi>>(
          interpolated_boundary_variables);
  CHECK_ITERABLE_CUSTOM_APPROX(get(kg_pi_nodal), get(interpolated_pi).data(),
                               angular_derivative_approx);
}

// This function tests `KleinGordonWorldtubeH5BufferUpdater`, which handles
// worldtube data of the Klein-Gordon system for CCE.
// The testing procedure involves the creation of synthetic modal data for the
// scalar field and its first time derivative, followed by the storage of this
// data in an HDF5 file. Subsequently, the `KleinGordonWorldtubeH5BufferUpdater`
// is used to read the worldtube data and verify that the loaded data matches
// the originally generated data.
//
// The test involves three buffer updaters
// (1) buffer_updater: a `KleinGordonWorldtubeH5BufferUpdater` object, built
//     from a HDF5 file to be written.
// (2) serialized_and_deserialized_updater: serialized-deserialized
//     buffer_updater.
// (3) dummy_buffer_updater: an object of `KleinGordonDummyBufferUpdater`
//     defined above.
//
// In the test, we treat `dummy_buffer_updater` as a reference object, whose
// stored modal data are to be compared with. For `buffer_updater` and
// `serialized_and_deserialized_updater`, we check:
//
// (1) The extraction radius is correctly retrieved.
// (2) The time stamps are the same as the ones we generate.
// (3) The worldtube data are the same as `dummy_buffer_updater`.
template <typename Generator>
void test_klein_gordon_worldtube_buffer_updater(
    const gsl::not_null<Generator*> gen,
    const bool extraction_radius_in_filename) {
  UniformCustomDistribution<double> value_dist{0.1, 0.5};

  const double extraction_radius = 100.0;

  const double frequency = 0.1 * value_dist(*gen);
  const double amplitude = 0.1 * value_dist(*gen);
  const double target_time = 50.0 * value_dist(*gen);

  const size_t buffer_size = 4;
  const size_t interpolator_length = 3;
  const size_t file_l_max = 8;
  const size_t computation_l_max = 10;

  const std::string filename = extraction_radius_in_filename
                                   ? "KgBoundaryDataH5Test_CceR0100.h5"
                                   : "KgBoundaryDataH5Test.h5";
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  // Generate fake modal data and write to `filename`
  {
    // scoped to close the file
    Scalar<ComplexModalVector> kg_psi_modal;
    Scalar<ComplexModalVector> kg_pi_modal;
    Scalar<DataVector> kg_psi_nodal;
    Scalar<DataVector> kg_pi_nodal;

    Cce::WorldtubeModeRecorder recorder{file_l_max, filename};
    for (size_t t = 0; t < 20; ++t) {
      const double time = 0.01 * static_cast<double>(t) + target_time - 0.1;

      TestHelpers::create_fake_time_varying_klein_gordon_data(
          make_not_null(&kg_psi_modal), make_not_null(&kg_pi_modal),
          make_not_null(&kg_psi_nodal), make_not_null(&kg_pi_nodal),
          extraction_radius, amplitude, frequency, time, file_l_max);

      recorder.append_modal_data<0>("/KGPsi", time, get(kg_psi_modal));
      recorder.append_modal_data<0>("/dtKGPsi", time, get(kg_pi_modal));
    }
  }

  // Create a `KleinGordonWorldtubeH5BufferUpdater` object `buffer_updater`
  // from the HDF5 file `filename` written above.
  // Then examine its extraction radius and time stamps.
  auto buffer_updater =
      extraction_radius_in_filename
          ? KleinGordonWorldtubeH5BufferUpdater{filename}
          : KleinGordonWorldtubeH5BufferUpdater{filename, extraction_radius};

  size_t time_span_start = 0;
  size_t time_span_end = 0;
  Variables<klein_gordon_input_tags> coefficients_buffers_from_file{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  buffer_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_file),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size);

  const auto& time_buffer = buffer_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer[i] == approx(target_time - 0.1 + 0.01 * i));
  }
  CHECK(buffer_updater.get_extraction_radius() == 100.0);

  // Test the `pup` function of `KleinGordonWorldtubeH5BufferUpdater`.
  // Serialize and deserialize `buffer_updater` and repeat the checks above.
  auto serialized_and_deserialized_updater =
      serialize_and_deserialize(buffer_updater);

  Variables<klein_gordon_input_tags> coefficients_buffers_from_serialized{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  size_t time_span_start_from_serialized = 0;
  size_t time_span_end_from_serialized = 0;
  serialized_and_deserialized_updater.update_buffers_for_time(
      make_not_null(&coefficients_buffers_from_serialized),
      make_not_null(&time_span_start_from_serialized),
      make_not_null(&time_span_end_from_serialized), target_time,
      computation_l_max, interpolator_length, buffer_size);

  const auto& time_buffer_from_serialized =
      serialized_and_deserialized_updater.get_time_buffer();
  for (size_t i = 0; i < time_buffer.size(); ++i) {
    CHECK(time_buffer_from_serialized[i] ==
          approx(target_time - 0.1 + 0.01 * i));
  }
  CHECK(serialized_and_deserialized_updater.get_extraction_radius() == 100.0);

  // Compare the modal data of `buffer_updater` and
  // `serialized_and_deserialized_updater`, which goes through the write-read
  // process, with what we generated earilier. The fiducial modal data are
  // stored in an object of `KleinGordonDummyBufferUpdater`.
  time_span_start = 0;
  time_span_end = 0;
  const KleinGordonDummyBufferUpdater dummy_buffer_updater{
      time_buffer, extraction_radius, amplitude, frequency, computation_l_max};
  Variables<klein_gordon_input_tags> expected_coefficients_buffers{
      (buffer_size + 2 * interpolator_length) * square(computation_l_max + 1)};
  dummy_buffer_updater.update_buffers_for_time(
      make_not_null(&expected_coefficients_buffers),
      make_not_null(&time_span_start), make_not_null(&time_span_end),
      target_time, computation_l_max, interpolator_length, buffer_size);

  Approx modal_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e3)
          .scale(1.0);

  tmpl::for_each<klein_gordon_input_tags>(
      [&expected_coefficients_buffers, &coefficients_buffers_from_file,
       &coefficients_buffers_from_serialized, &modal_approx](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        INFO(db::tag_name<tag>());
        const auto& expected_coefs = get<tag>(expected_coefficients_buffers);
        const auto& kg_coefs = get<tag>(coefficients_buffers_from_file);
        CHECK_ITERABLE_CUSTOM_APPROX(expected_coefs, kg_coefs, modal_approx);
        const auto& serialized_kg_coefs =
            get<tag>(coefficients_buffers_from_serialized);
        CHECK_ITERABLE_CUSTOM_APPROX(expected_coefs, serialized_kg_coefs,
                                     modal_approx);
      });

  // Finally remove the generated file
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ReadKleinGordonBoundaryDataH5",
                  "[Unit][Cce]") {
  register_derived_classes_with_charm<
      Cce::WorldtubeBufferUpdater<klein_gordon_input_tags>>();
  register_derived_classes_with_charm<Cce::WorldtubeDataManager<
      Cce::Tags::klein_gordon_worldtube_boundary_tags>>();
  register_derived_classes_with_charm<intrp::SpanInterpolator>();
  MAKE_GENERATOR(gen);
  {
    INFO("Testing Klein-Gordon buffer updaters");
    test_klein_gordon_worldtube_buffer_updater(make_not_null(&gen), true);
    test_klein_gordon_worldtube_buffer_updater(make_not_null(&gen), false);
  }
  {
    INFO("Testing Klein-Gordon data manager");
    test_klein_gordon_data_manager_with_dummy_buffer_updater(
        make_not_null(&gen));
  }
}
}  // namespace Cce
