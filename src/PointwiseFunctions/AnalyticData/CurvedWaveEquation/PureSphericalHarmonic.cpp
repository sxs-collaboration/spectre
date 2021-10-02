// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"

#include <complex>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshInterpolation.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace CurvedScalarWave::AnalyticData {

PureSphericalHarmonic::PureSphericalHarmonic(const double radius,
                                             const double width,
                                             std::pair<size_t, int> mode,
                                             const Options::Context& context)
    : radius_(radius), width_sq_(width * width), mode_{std::move(mode)} {
  if (abs(mode_.second) > static_cast<int>(mode_.first)) {
    PARSE_ERROR(
        context,
        "The absolute value of the m_mode must be less than or equal to the "
        "l-mode but the m-mode is "
            << mode_.second << " and the l-mode is " << mode_.first);
  }
  if (radius_ <= 0.) {
    PARSE_ERROR(context,
                "The radius must be greater than 0 but is " << radius_);
  }
  if (width <= 0.) {
    PARSE_ERROR(context, "The width must be greater than 0 but is " << width);
  }
}

tuples::TaggedTuple<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                    CurvedScalarWave::Psi>
PureSphericalHarmonic::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
               CurvedScalarWave::Psi> /*meta*/) const {
  Scalar<DataVector> pi{get(magnitude(x)) - radius_};
  get(pi) = exp(-get(pi) * get(pi) / width_sq_);
  const Spectral::Swsh::SpinWeightedSphericalHarmonic spherical_harmonic(
      0, mode_.first, mode_.second);
  const auto theta = atan2(hypot(x[0], x[1]), x[2]);
  const auto phi = atan2(x[1], x[0]);
  get(pi) *= real(spherical_harmonic.evaluate(theta, phi, sin(theta / 2.),
                                              cos(theta / 2.)));
  return tuples::TaggedTuple<CurvedScalarWave::Pi, CurvedScalarWave::Phi<3>,
                             CurvedScalarWave::Psi>{
      std::move(pi), make_with_value<tnsr::i<DataVector, 3>>(x, 0.),
      make_with_value<Scalar<DataVector>>(x, 0.)};
}

void PureSphericalHarmonic::pup(PUP::er& p) {
  p | radius_;
  p | width_sq_;
  p | mode_;
}

bool operator==(const PureSphericalHarmonic& lhs,
                const PureSphericalHarmonic& rhs) {
  return lhs.radius_ == rhs.radius_ and lhs.width_sq_ == rhs.width_sq_ and
         lhs.mode_ == rhs.mode_;
}
bool operator!=(const PureSphericalHarmonic& lhs,
                const PureSphericalHarmonic& rhs) {
  return not(lhs == rhs);
}
}  // namespace CurvedScalarWave::AnalyticData
