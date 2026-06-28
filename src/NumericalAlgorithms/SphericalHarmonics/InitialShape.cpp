// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/InitialShape.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/Tensor/IndexType.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame

namespace ylm::InitialShapes {
template <typename Frame>
Sphere<Frame>::Sphere(std::array<double, 3> center, const double radius)
    : center_(std::move(center)), radius_(radius) {}

template <typename Frame>
Sphere<Frame>::Sphere(CkMigrateMessage* msg) : InitialShape<Frame>(msg) {}

template <typename Frame>
Strahlkorper<Frame> Sphere<Frame>::strahlkorper(
    const size_t l_max, const Options::Context& /*context*/) const {
  return Strahlkorper<Frame>{l_max, radius_, center_};
}

template <typename Frame>
void Sphere<Frame>::pup(PUP::er& p) {
  p | center_;
  p | radius_;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class Sphere<FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
}  // namespace ylm::InitialShapes

template <typename Frame>
PUP::able::PUP_ID ylm::InitialShapes::Sphere<Frame>::my_PUP_ID = 0;  // NOLINT
