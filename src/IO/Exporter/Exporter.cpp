// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/Exporter.h"
#include "IO/Exporter/Exporter.hpp"

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Error message of the last failed call to the C interface on this thread
std::string& last_error() {
  thread_local std::string message{};
  return message;
}
}  // namespace

extern "C" int spectre_interpolate_to_points(
    double* const* const out, const char* const* const volume_files,
    const size_t num_volume_files, const char* const subfile_name,
    const int observation_step, const char* const* const tensor_components,
    const size_t num_tensor_components,
    const double* const* const target_points, const size_t num_dimensions,
    const size_t num_points, const bool extrapolate_into_excisions,
    const bool error_on_missing_points, const int num_threads) {
  last_error().clear();
  // Exceptions must not propagate into the calling C code
  try {
    const auto volume_files_span =
        gsl::make_span(volume_files, num_volume_files);
    const auto tensor_components_span =
        gsl::make_span(tensor_components, num_tensor_components);
    const auto out_span = gsl::make_span(out, num_tensor_components);
    const auto target_points_span =
        gsl::make_span(target_points, num_dimensions);
    const auto impl = [&]<size_t Dim>() {
      // Reference the caller's memory for the target points and the result so
      // no data is copied
      tnsr::I<DataVector, Dim, Frame::Inertial> target_points_dv{};
      for (size_t d = 0; d < Dim; ++d) {
        target_points_dv.get(d).set_data_ref(
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            const_cast<double*>(gsl::at(target_points_span, d)), num_points);
      }
      std::vector<DataVector> result(num_tensor_components);
      for (size_t i = 0; i < num_tensor_components; ++i) {
        result[i].set_data_ref(gsl::at(out_span, i), num_points);
      }
      spectre::Exporter::interpolate_to_points(
          make_not_null(&result),
          std::vector<std::string>(volume_files_span.begin(),
                                   volume_files_span.end()),
          std::string(subfile_name),
          spectre::Exporter::ObservationStep(observation_step),
          std::vector<std::string>(tensor_components_span.begin(),
                                   tensor_components_span.end()),
          target_points_dv, extrapolate_into_excisions, error_on_missing_points,
          num_threads <= 0 ? std::nullopt : std::optional<size_t>(num_threads));
    };
    if (num_dimensions == 1) {
      impl.template operator()<1>();
    } else if (num_dimensions == 2) {
      impl.template operator()<2>();
    } else if (num_dimensions == 3) {
      impl.template operator()<3>();
    } else {
      ERROR_NO_TRACE("Unsupported number of dimensions: "
                     << num_dimensions << ". Must be 1, 2, or 3.");
    }
    return 0;
  } catch (const std::exception& error) {
    last_error() = error.what();
  } catch (...) {
    last_error() = "Unknown error.";
  }
  return 1;
}

extern "C" const char* spectre_exporter_last_error() {
  return last_error().c_str();
}

namespace spectre::Exporter {

template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    const bool extrapolate_into_excisions, const bool error_on_missing_points,
    const std::optional<size_t> num_threads) {
  tnsr::I<DataVector, Dim, Frame::Inertial> target_points_dv{};
  for (size_t d = 0; d < Dim; ++d) {
    target_points_dv.get(d).set_data_ref(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(gsl::at(target_points, d).data()),
        gsl::at(target_points, d).size());
  }
  std::vector<std::vector<double>> result{};
  interpolate_to_points(make_not_null(&result), volume_files_or_glob,
                        subfile_name, observation, tensor_components,
                        target_points_dv, extrapolate_into_excisions,
                        error_on_missing_points, num_threads);
  return result;
}

// Generate instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::vector<std::vector<double>> interpolate_to_points<DIM(data)>( \
      const std::variant<std::vector<std::string>, std::string>&              \
          volume_files_or_glob,                                               \
      const std::string& subfile_name, const ObservationVariant& observation, \
      const std::vector<std::string>& tensor_components,                      \
      const std::array<std::vector<double>, DIM(data)>& target_points,        \
      bool extrapolate_into_excisions, bool error_on_missing_points,          \
      std::optional<size_t> num_threads);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter
