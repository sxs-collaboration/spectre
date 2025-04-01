// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/Exporter.hpp"

#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace spectre::Exporter {

template <size_t Dim>
std::vector<std::vector<double>> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const std::array<std::vector<double>, Dim>& target_points,
    const bool extrapolate_into_excisions,
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
                        num_threads);
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
      bool extrapolate_into_excisions, std::optional<size_t> num_threads);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace spectre::Exporter
