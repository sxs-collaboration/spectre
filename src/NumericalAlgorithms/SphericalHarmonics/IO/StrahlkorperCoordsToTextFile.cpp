// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/StrahlkorperCoordsToTextFile.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Inertial;
struct Distorted;
struct Grid;
}  // namespace Frame

namespace ylm {
template <typename Frame>
void write_strahlkorper_coords_to_text_file(
    const Strahlkorper<Frame>& strahlkorper,
    const std::string& output_file_name, const AngularOrdering ordering,
    const bool overwrite_file) {
  if (not overwrite_file and
      file_system::check_if_file_exists(output_file_name)) {
    ERROR_NO_TRACE("The output file " << output_file_name
                                      << " already exists.");
  }

  tnsr::I<DataVector, 3, Frame> cartesian_coords =
      ylm::cartesian_coords(strahlkorper);

  // Cce expects coordinates in a different order than a typical Strahlkorper
  if (ordering == AngularOrdering::Cce) {
    const auto physical_extents =
        strahlkorper.ylm_spherepack().physical_extents();
    auto transposed_coords =
        tnsr::I<DataVector, 3, Frame>(get<0>(cartesian_coords).size());
    for (size_t i = 0; i < 3; ++i) {
      transpose(make_not_null(&transposed_coords.get(i)),
                cartesian_coords.get(i), physical_extents[0],
                physical_extents[1]);
    }

    cartesian_coords = std::move(transposed_coords);
  }

  std::ofstream output_file(output_file_name);
  output_file << std::fixed
              << std::setprecision(std::numeric_limits<double>::digits10 + 4)
              << std::scientific;

  const size_t num_points = get<0>(cartesian_coords).size();
  for (size_t i = 0; i < num_points; i++) {
    output_file << get<0>(cartesian_coords)[i] << " "
                << get<1>(cartesian_coords)[i] << " "
                << get<2>(cartesian_coords)[i] << std::endl;
  }
}

void write_strahlkorper_coords_to_text_file(const double radius,
                                            const size_t l_max,
                                            const std::array<double, 3>& center,
                                            const std::string& output_file_name,
                                            const AngularOrdering ordering,
                                            const bool overwrite_file) {
  const Strahlkorper<Frame::Inertial> strahlkorper{l_max, radius, center};
  write_strahlkorper_coords_to_text_file(strahlkorper, output_file_name,
                                         ordering, overwrite_file);
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template void write_strahlkorper_coords_to_text_file(                    \
      const Strahlkorper<FRAME(data)>& strahlkorper,                       \
      const std::string& output_file_name, const AngularOrdering ordering, \
      const bool overwrite_file);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef INSTANTIATE
#undef FRAME
}  // namespace ylm
