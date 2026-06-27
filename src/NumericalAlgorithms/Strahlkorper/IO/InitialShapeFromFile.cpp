// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Strahlkorper/IO/InitialShapeFromFile.hpp"

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>

#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame

namespace ylm::InitialShapes {
template <typename Frame>
FromFile<Frame>::FromFile(std::string h5_filename, std::string subfile_name,
                          const double time, const double time_epsilon,
                          const bool check_frame)
    : h5_filename_(std::move(h5_filename)),
      subfile_name_(std::move(subfile_name)),
      time_(time),
      time_epsilon_(time_epsilon),
      check_frame_(check_frame) {}

template <typename Frame>
FromFile<Frame>::FromFile(CkMigrateMessage* msg) : InitialShape<Frame>(msg) {}

template <typename Frame>
Strahlkorper<Frame> FromFile<Frame>::strahlkorper(
    const size_t l_max, const Options::Context& context) const {
  return Strahlkorper<Frame>{l_max,         h5_filename_, subfile_name_, time_,
                             time_epsilon_, check_frame_, context};
}

template <typename Frame>
void FromFile<Frame>::pup(PUP::er& p) {
  p | h5_filename_;
  p | subfile_name_;
  p | time_;
  p | time_epsilon_;
  p | check_frame_;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class FromFile<FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
}  // namespace ylm::InitialShapes

template <typename Frame>
PUP::able::PUP_ID ylm::InitialShapes::FromFile<Frame>::my_PUP_ID = 0;  // NOLINT
