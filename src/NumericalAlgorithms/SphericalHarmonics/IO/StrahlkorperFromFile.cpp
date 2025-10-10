// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

#include <string>

#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "Options/Context.hpp"

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame

namespace ylm {
template <typename Frame>
Strahlkorper<Frame>::Strahlkorper(const size_t l_max,
                                  const std::string& h5_filename,
                                  const std::string& subfile_name,
                                  const double time, const double time_epsilon,
                                  const bool check_frame,
                                  const Options::Context& /*context*/)
    : Strahlkorper(
          l_max, l_max,
          ylm::read_surface_ylm_single_time<Frame>(
              h5_filename, subfile_name, time, time_epsilon, check_frame)) {}

template class Strahlkorper<Frame::Inertial>;
template class Strahlkorper<Frame::Grid>;
template class Strahlkorper<Frame::Distorted>;
}  // namespace ylm
