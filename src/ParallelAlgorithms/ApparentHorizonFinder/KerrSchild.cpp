// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/KerrSchild.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <utility>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame

namespace ah::InitialShapes {
template <typename Frame>
KerrSchild<Frame>::KerrSchild(std::array<double, 3> center, const double mass,
                              std::array<double, 3> spin)
    : center_(std::move(center)), mass_(mass), spin_(std::move(spin)) {}

template <typename Frame>
KerrSchild<Frame>::KerrSchild(CkMigrateMessage* msg)
    : ylm::InitialShape<Frame>(msg) {}

template <typename Frame>
ylm::Strahlkorper<Frame> KerrSchild<Frame>::strahlkorper(
    const size_t l_max, const Options::Context& context) const {
  if (mass_ <= 0.0) {
    PARSE_ERROR(context, "KerrSchild expects Mass > 0, not " << mass_);
  }
  if (magnitude(spin_) > 1.0) {
    PARSE_ERROR(context,
                "KerrSchild expects |Spin| <= 1, not " << magnitude(spin_));
  }
  const auto ylm = ylm::Spherepack{l_max, l_max};
  return ylm::Strahlkorper<Frame>{l_max, l_max,
                                  get(gr::Solutions::kerr_horizon_radius(
                                      ylm.theta_phi_points(), mass_, spin_)),
                                  center_};
}

template <typename Frame>
void KerrSchild<Frame>::pup(PUP::er& p) {
  p | center_;
  p | mass_;
  p | spin_;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template class KerrSchild<FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (::Frame::Grid, ::Frame::Distorted, ::Frame::Inertial))

#undef FRAME
#undef INSTANTIATE
}  // namespace ah::InitialShapes

template <typename Frame>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID ah::InitialShapes::KerrSchild<Frame>::my_PUP_ID =
    0;  // NOLINT
