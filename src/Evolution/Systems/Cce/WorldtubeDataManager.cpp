// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeDataManager.hpp"

#include <complex>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

namespace detail {
template <typename InputTags>
void set_non_pupped_members(
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end,
    const gsl::not_null<Variables<InputTags>*> coefficients_buffers,
    const gsl::not_null<Variables<InputTags>*> interpolated_coefficients,
    const size_t buffer_depth, const size_t interpolator_length,
    const size_t l_max) {
  *time_span_start = 0;
  *time_span_end = 0;
  const size_t size_of_buffer =
      square(l_max + 1) * (buffer_depth + 2 * interpolator_length);
  *coefficients_buffers = Variables<InputTags>{size_of_buffer};
  *interpolated_coefficients = Variables<InputTags>{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)};
}

template <typename InputTags>
void initialize_buffers(
    const gsl::not_null<size_t*> buffer_depth,
    const gsl::not_null<Variables<InputTags>*> coefficients_buffers,
    const size_t buffer_size, const size_t interpolator_length,
    const size_t l_max) {
  if (UNLIKELY(buffer_size < 2 * interpolator_length)) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolator. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the specified Interpolator requests too many "
        "points");
  }
  // This will actually change the buffer depth in the case where the buffer
  // depth passed to the constructor is too large for the worldtube file size.
  // In that case, the worldtube data wouldn't be able to fill the buffer, so
  // here we shrink the buffer depth down to be no larger than the length of the
  // worldtube file.
  if (UNLIKELY(buffer_size < 2 * interpolator_length + (*buffer_depth))) {
    *buffer_depth = buffer_size - 2 * interpolator_length;
  }

  const size_t size_of_buffer =
      square(l_max + 1) * (*buffer_depth + 2 * interpolator_length);
  *coefficients_buffers = Variables<InputTags>{size_of_buffer};
}

template <typename InputTags, typename OutputTags>
void populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<OutputTags>*> boundary_data_variables,
    const gsl::not_null<Variables<InputTags>*> interpolated_coefficients,
    const gsl::not_null<Variables<InputTags>*> coefficients_buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock, const double time,
    const std::unique_ptr<intrp::SpanInterpolator>& interpolator,
    const std::unique_ptr<WorldtubeBufferUpdater<InputTags>>& buffer_updater,
    const size_t l_max, const size_t buffer_depth) {
  {
    const std::lock_guard hold_lock(*hdf5_lock);
    buffer_updater->update_buffers_for_time(
        coefficients_buffers, time_span_start, time_span_end, time, l_max,
        interpolator->required_number_of_points_before_and_after(),
        buffer_depth);
  }

  auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator->required_number_of_points_before_and_after(),
      *time_span_start, *time_span_end, buffer_updater->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = (*time_span_end) - (*time_span_start);
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column = [&time, &time_points, &buffer_span_size,
                                  &interpolation_time_span,
                                  &interpolation_span_size, &time_span_start,
                                  &interpolator](auto data, size_t column) {
    const auto interp_val = interpolator->interpolate(
        gsl::span<const double>(time_points.data(), time_points.size()),
        gsl::span<const std::complex<double>>(
            data + column * (buffer_span_size) +
                (interpolation_time_span.first - (*time_span_start)),
            interpolation_span_size),
        time);
    return interp_val;
  };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.

  for (const auto libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max)) {
    tmpl::for_each<InputTags>([&libsharp_mode, &interpolate_from_column,
                               &interpolated_coefficients, &l_max,
                               &coefficients_buffers](auto tag_v) {
      using tag = typename decltype(tag_v)::type;
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode,
          make_not_null(&get(get<tag>(*interpolated_coefficients))), 0,
          interpolate_from_column(
              get(get<tag>(*coefficients_buffers)).data().data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max, libsharp_mode.l, static_cast<int>(libsharp_mode.m))),
          interpolate_from_column(
              get(get<tag>(*coefficients_buffers)).data().data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max, libsharp_mode.l, -static_cast<int>(libsharp_mode.m))));
    });
  }
  // just inverse transform the 'direct' tags
  tmpl::for_each<
      tmpl::transform<InputTags, tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
      [&boundary_data_variables, &interpolated_coefficients,
       &l_max](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        Spectral::Swsh::inverse_swsh_transform(
            l_max, 1,
            make_not_null(
                &get(get<Tags::BoundaryValue<tag>>(*boundary_data_variables))),
            get(get<Spectral::Swsh::Tags::SwshTransform<tag>>(
                *interpolated_coefficients)));
      });
}
}  // namespace detail

MetricWorldtubeDataManager::MetricWorldtubeDataManager(
    std::unique_ptr<WorldtubeBufferUpdater<input_tags>> buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator,
    const bool fix_spec_normalization)
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      fix_spec_normalization_{fix_spec_normalization},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  detail::initialize_buffers<input_tags>(
      make_not_null(&buffer_depth_), make_not_null(&coefficients_buffers_),
      buffer_updater_->get_time_buffer().size(),
      interpolator_->required_number_of_points_before_and_after(), l_max);
}

bool MetricWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  {
    const std::lock_guard hold_lock(*hdf5_lock);
    buffer_updater_->update_buffers_for_time(
        make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
        make_not_null(&time_span_end_), time, l_max_,
        interpolator_->required_number_of_points_before_and_after(),
        buffer_depth_);
  }
  const auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  const DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column = [&time, &time_points, &buffer_span_size,
                                  &interpolation_time_span,
                                  &interpolation_span_size,
                                  this](auto data, const size_t column) {
    auto interp_val = interpolator_->interpolate(
        gsl::span<const double>(time_points.data(), time_points.size()),
        gsl::span<const std::complex<double>>(
            data + column * buffer_span_size +
                (interpolation_time_span.first - time_span_start_),
            interpolation_span_size),
        time);
    return interp_val;
  };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.

  // we'll just use this buffer to reference into the actual data to satisfy
  // the swsh interface requirement that the spin-weight be labelled with
  // `SpinWeighted`
  SpinWeighted<ComplexModalVector, 0> spin_weighted_buffer;
  for (const auto libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        tmpl::for_each<Tags::detail::apply_derivs_t<
            Tags::detail::SpatialMetric<ComplexModalVector>>>(
            [this, &i, &j, &libsharp_mode, &interpolate_from_column,
             &spin_weighted_buffer](auto tag_v) {
              using tag = typename decltype(tag_v)::type;
              spin_weighted_buffer.set_data_ref(
                  get<tag>(interpolated_coefficients_).get(i, j).data(),
                  Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
              Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                  libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                  interpolate_from_column(
                      get<tag>(coefficients_buffers_).get(i, j).data(),
                      Spectral::Swsh::goldberg_mode_index(
                          l_max_, libsharp_mode.l,
                          static_cast<int>(libsharp_mode.m))),
                  interpolate_from_column(
                      get<tag>(coefficients_buffers_).get(i, j).data(),
                      Spectral::Swsh::goldberg_mode_index(
                          l_max_, libsharp_mode.l,
                          -static_cast<int>(libsharp_mode.m))));
            });
      }
      tmpl::for_each<Tags::detail::apply_derivs_t<
          Tags::detail::Shift<ComplexModalVector>>>(
          [this, &i, &libsharp_mode, &interpolate_from_column,
           &spin_weighted_buffer](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            spin_weighted_buffer.set_data_ref(
                get<tag>(interpolated_coefficients_).get(i).data(),
                Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
            Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        static_cast<int>(libsharp_mode.m))),
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        -static_cast<int>(libsharp_mode.m))));
          });
    }
    tmpl::for_each<
        Tags::detail::apply_derivs_t<Tags::detail::Lapse<ComplexModalVector>>>(
        [this, &libsharp_mode, &interpolate_from_column,
         &spin_weighted_buffer](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          spin_weighted_buffer.set_data_ref(
              get(get<tag>(interpolated_coefficients_)).data(),
              Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
          Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
              libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      static_cast<int>(libsharp_mode.m))),
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      -static_cast<int>(libsharp_mode.m))));
        });
  }

  // At this point, we have a collection of 9 tensors of libsharp
  // coefficients. This is what the boundary data calculation utility takes
  // as an input, so we now hand off the control flow to the boundary and
  // gauge transform utility
  // This relies on the ordering of tags in the `input_tags` type alias
  const auto create_boundary_data = [&](const auto&... tags) {
    if (not buffer_updater_->has_version_history() and
        fix_spec_normalization_) {
      create_bondi_boundary_data_from_unnormalized_spec_modes(
          boundary_data_variables,
          get<tmpl::type_from<std::decay_t<decltype(tags)>>>(
              interpolated_coefficients_)...,
          buffer_updater_->get_extraction_radius(), l_max_);
    } else {
      create_bondi_boundary_data(
          boundary_data_variables,
          get<tmpl::type_from<std::decay_t<decltype(tags)>>>(
              interpolated_coefficients_)...,
          buffer_updater_->get_extraction_radius(), l_max_);
    }
  };

  tmpl::as_pack<input_tags>(create_boundary_data);

  return true;
}

std::unique_ptr<WorldtubeDataManager<
    Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>>
MetricWorldtubeDataManager::get_clone() const {
  return std::make_unique<MetricWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone(), fix_spec_normalization_);
}

std::pair<size_t, size_t> MetricWorldtubeDataManager::get_time_span() const {
  return std::make_pair(time_span_start_, time_span_end_);
}

void MetricWorldtubeDataManager::pup(PUP::er& p) {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  p | fix_spec_normalization_;
  if (p.isUnpacking()) {
    detail::set_non_pupped_members<input_tags>(
        make_not_null(&time_span_start_), make_not_null(&time_span_end_),
        make_not_null(&coefficients_buffers_),
        make_not_null(&interpolated_coefficients_), buffer_depth_,
        interpolator_->required_number_of_points_before_and_after(), l_max_);
  }
}

BondiWorldtubeDataManager::BondiWorldtubeDataManager(
    std::unique_ptr<
        WorldtubeBufferUpdater<Tags::worldtube_boundary_tags_for_writing<
            Spectral::Swsh::Tags::SwshTransform>>>
        buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator)
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  detail::initialize_buffers<Tags::worldtube_boundary_tags_for_writing<
      Spectral::Swsh::Tags::SwshTransform>>(
      make_not_null(&buffer_depth_), make_not_null(&coefficients_buffers_),
      buffer_updater_->get_time_buffer().size(),
      interpolator_->required_number_of_points_before_and_after(), l_max);
}

bool BondiWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }

  detail::populate_hypersurface_boundary_data<
      Tags::worldtube_boundary_tags_for_writing<
          Spectral::Swsh::Tags::SwshTransform>,
      Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>(
      boundary_data_variables, make_not_null(&interpolated_coefficients_),
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), hdf5_lock, time, interpolator_,
      buffer_updater_, l_max_, buffer_depth_);

  const auto& du_r = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
      *boundary_data_variables));
  const auto& bondi_r =
      get(get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_variables));

  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*boundary_data_variables)) =
      du_r / bondi_r;

  // there's only a couple of tags desired by the core computation that aren't
  // stored in the bondi format, so we perform the remaining computation
  // in-line here.
  const auto& du_bondi_j = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
      *boundary_data_variables));
  const auto& dr_bondi_j = get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
      *boundary_data_variables));
  get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables)) =
      du_bondi_j + du_r * dr_bondi_j;

  const auto& bondi_j =
      get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables));
  const auto& bondi_beta =
      get(get<Tags::BoundaryValue<Tags::BondiBeta>>(*boundary_data_variables));
  const auto& bondi_q =
      get(get<Tags::BoundaryValue<Tags::BondiQ>>(*boundary_data_variables));
  const auto& bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));
  get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
      *boundary_data_variables)) =
      exp(2.0 * bondi_beta.data()) / square(bondi_r.data()) *
      (bondi_k.data() * bondi_q.data() - bondi_j.data() * conj(bondi_q.data()));

  populate_boundary_du_dy_j(boundary_data_variables, time);
  return true;
}

void BondiWorldtubeDataManager::populate_boundary_du_dy_j(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
    const double time) const {
  // The previous detail::populate_hypersurface_boundary_data call already
  // refreshed time_span_start_ / time_span_end_ for the current target time.
  // The interpolation sub-window we use for the time derivative must match the
  // one used for value interpolation above.
  const auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t offset_in_buffer =
      interpolation_time_span.first - time_span_start_;

  const DataVector time_points{
      buffer_updater_->get_time_buffer().data() +  // NOLINT
          interpolation_time_span.first,
      interpolation_span_size};

  const size_t number_of_libsharp_coefficients =
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_);
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max_);

  SpinWeighted<ComplexModalVector, 0> r_libsharp_modes{
      number_of_libsharp_coefficients};
  SpinWeighted<ComplexModalVector, 2> dr_j_libsharp_modes{
      number_of_libsharp_coefficients};
  SpinWeighted<ComplexDataVector, 0> r_nodal{number_of_angular_points};
  SpinWeighted<ComplexDataVector, 2> dr_j_nodal{number_of_angular_points};

  // dy_j over time, laid out so the nodal slice at buffer time `ti` lives in
  // [ti * num_points, (ti + 1) * num_points).
  ComplexDataVector dy_j_over_time{interpolation_span_size *
                                   number_of_angular_points};

  const auto& r_modes_buffer =
      get(get<Spectral::Swsh::Tags::SwshTransform<Tags::BondiR>>(
              coefficients_buffers_))
          .data();
  const auto& dr_j_modes_buffer =
      get(get<Spectral::Swsh::Tags::SwshTransform<Tags::Dr<Tags::BondiJ>>>(
              coefficients_buffers_))
          .data();

  for (size_t ti = 0; ti < interpolation_span_size; ++ti) {
    const size_t time_index_in_buffer = offset_in_buffer + ti;
    r_libsharp_modes.data() = 0.0;
    dr_j_libsharp_modes.data() = 0.0;
    for (const auto libsharp_mode :
         Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
      const size_t plus_m_goldberg_index = Spectral::Swsh::goldberg_mode_index(
          l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m));
      const size_t minus_m_goldberg_index = Spectral::Swsh::goldberg_mode_index(
          l_max_, libsharp_mode.l, -static_cast<int>(libsharp_mode.m));
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode, make_not_null(&r_libsharp_modes), 0,
          r_modes_buffer[(plus_m_goldberg_index * buffer_span_size) +
                         time_index_in_buffer],
          r_modes_buffer[(minus_m_goldberg_index * buffer_span_size) +
                         time_index_in_buffer]);
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode, make_not_null(&dr_j_libsharp_modes), 0,
          dr_j_modes_buffer[(plus_m_goldberg_index * buffer_span_size) +
                            time_index_in_buffer],
          dr_j_modes_buffer[(minus_m_goldberg_index * buffer_span_size) +
                            time_index_in_buffer]);
    }
    Spectral::Swsh::inverse_swsh_transform(l_max_, 1, make_not_null(&r_nodal),
                                           r_libsharp_modes);
    Spectral::Swsh::inverse_swsh_transform(
        l_max_, 1, make_not_null(&dr_j_nodal), dr_j_libsharp_modes);

    ComplexDataVector dy_j_at_time{
        dy_j_over_time.data() + (ti * number_of_angular_points),  // NOLINT
        number_of_angular_points};
    dy_j_at_time = 0.5 * r_nodal.data() * dr_j_nodal.data();
  }

  auto& du_dy_j =
      get(get<Tags::BoundaryValue<Tags::Du<Tags::Dy<Tags::BondiJ>>>>(
              *boundary_data_variables))
          .data();
  ComplexDataVector dy_j_at_point{interpolation_span_size};
  for (size_t i = 0; i < number_of_angular_points; ++i) {
    for (size_t ti = 0; ti < interpolation_span_size; ++ti) {
      dy_j_at_point[ti] = dy_j_over_time[(ti * number_of_angular_points) + i];
    }
    du_dy_j[i] = interpolator_->derivative(
        gsl::span<const double>(time_points.data(), time_points.size()),
        gsl::span<const std::complex<double>>(dy_j_at_point.data(),
                                              dy_j_at_point.size()),
        time);
  }
}

std::unique_ptr<WorldtubeDataManager<
    Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>>
BondiWorldtubeDataManager::get_clone() const {
  return std::make_unique<BondiWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone());
}

std::pair<size_t, size_t> BondiWorldtubeDataManager::get_time_span() const {
  return std::make_pair(time_span_start_, time_span_end_);
}

void BondiWorldtubeDataManager::pup(PUP::er& p) {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    detail::set_non_pupped_members<Tags::worldtube_boundary_tags_for_writing<
        Spectral::Swsh::Tags::SwshTransform>>(
        make_not_null(&time_span_start_), make_not_null(&time_span_end_),
        make_not_null(&coefficients_buffers_),
        make_not_null(&interpolated_coefficients_), buffer_depth_,
        interpolator_->required_number_of_points_before_and_after(), l_max_);
  }
}

KleinGordonWorldtubeDataManager::KleinGordonWorldtubeDataManager(
    std::unique_ptr<WorldtubeBufferUpdater<klein_gordon_input_tags>>
        buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator)
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  detail::initialize_buffers<klein_gordon_input_tags>(
      make_not_null(&buffer_depth_), make_not_null(&coefficients_buffers_),
      buffer_updater_->get_time_buffer().size(),
      interpolator_->required_number_of_points_before_and_after(), l_max);
}

bool KleinGordonWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<Tags::klein_gordon_worldtube_boundary_tags>*>
        boundary_data_variables,
    const double time,
    const gsl::not_null<Parallel::NodeLock*> hdf5_lock) const {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }

  detail::populate_hypersurface_boundary_data<
      klein_gordon_input_tags, Tags::klein_gordon_worldtube_boundary_tags>(
      boundary_data_variables, make_not_null(&interpolated_coefficients_),
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), hdf5_lock, time, interpolator_,
      buffer_updater_, l_max_, buffer_depth_);

  return true;
}

std::unique_ptr<
    WorldtubeDataManager<Tags::klein_gordon_worldtube_boundary_tags>>
KleinGordonWorldtubeDataManager::get_clone() const {
  return std::make_unique<KleinGordonWorldtubeDataManager>(
      buffer_updater_->get_clone(), l_max_, buffer_depth_,
      interpolator_->get_clone());
}

std::pair<size_t, size_t> KleinGordonWorldtubeDataManager::get_time_span()
    const {
  return std::make_pair(time_span_start_, time_span_end_);
}

void KleinGordonWorldtubeDataManager::pup(PUP::er& p) {
  p | buffer_updater_;
  p | time_span_start_;
  p | time_span_end_;
  p | l_max_;
  p | buffer_depth_;
  p | interpolator_;
  if (p.isUnpacking()) {
    detail::set_non_pupped_members<klein_gordon_input_tags>(
        make_not_null(&time_span_start_), make_not_null(&time_span_end_),
        make_not_null(&coefficients_buffers_),
        make_not_null(&interpolated_coefficients_), buffer_depth_,
        interpolator_->required_number_of_points_before_and_after(), l_max_);
  }
}

PUP::able::PUP_ID MetricWorldtubeDataManager::my_PUP_ID = 0;       // NOLINT
PUP::able::PUP_ID BondiWorldtubeDataManager::my_PUP_ID = 0;        // NOLINT
PUP::able::PUP_ID KleinGordonWorldtubeDataManager::my_PUP_ID = 0;  // NOLINT
}  // namespace Cce
