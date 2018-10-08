// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"           // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Parallel/PupStlCpp11.hpp"               // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
namespace Solutions {

RegularSphericalWave::RegularSphericalWave(
    std::unique_ptr<MathFunction<1>> profile) noexcept
    : profile_(std::move(profile)) {}

tuples::TaggedTuple<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>
RegularSphericalWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    const tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>,
                     ScalarWave::Psi> /*meta*/) const noexcept {
  const DataVector r = get(magnitude(x));
  // See class documentation for choice of cutoff
  const double r_cutoff = cbrt(std::numeric_limits<double>::epsilon());
  Scalar<DataVector> psi{r.size()};
  Scalar<DataVector> dpsi_dt{r.size()};
  tnsr::i<DataVector, 3> dpsi_dx{r.size()};
  for (size_t i = 0; i < r.size(); i++) {
    // Testing for r=0 here assumes a scale of order unity
    if (equal_within_roundoff(r[i], 0., r_cutoff, 1.)) {
      get(psi)[i] = 2. * profile_->first_deriv(-t);
      get(dpsi_dt)[i] = -2. * profile_->second_deriv(-t);
      for (size_t d = 0; d < 3; d++) {
        dpsi_dx.get(d)[i] = 0.;
      }
    } else {
      const auto F_out = profile_->operator()(r[i] - t);
      const auto F_in = profile_->operator()(-r[i] - t);
      const auto dF_out = profile_->first_deriv(r[i] - t);
      const auto dF_in = profile_->first_deriv(-r[i] - t);
      get(psi)[i] = (F_out - F_in) / r[i];
      get(dpsi_dt)[i] = (-dF_out + dF_in) / r[i];
      const double dpsi_dx_isotropic =
          (dF_out + dF_in - get(psi)[i]) / square(r[i]);
      for (size_t d = 0; d < 3; d++) {
        dpsi_dx.get(d)[i] = dpsi_dx_isotropic * x.get(d)[i];
      }
    }
  }
  tuples::TaggedTuple<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>
      variables{std::move(dpsi_dt), std::move(dpsi_dx), std::move(psi)};
  get<ScalarWave::Pi>(variables).get() *= -1.0;
  return variables;
}

tuples::TaggedTuple<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<3>>,
                    Tags::dt<ScalarWave::Psi>>
RegularSphericalWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    const tmpl::list<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<3>>,
                     Tags::dt<ScalarWave::Psi>> /*meta*/) const noexcept {
  const DataVector r = get(magnitude(x));
  // See class documentation for choice of cutoff
  const double r_cutoff = cbrt(std::numeric_limits<double>::epsilon());
  Scalar<DataVector> dpsi_dt{r.size()};
  Scalar<DataVector> d2psi_dt2{r.size()};
  tnsr::i<DataVector, 3> d2psi_dtdx{r.size()};
  for (size_t i = 0; i < r.size(); i++) {
    // Testing for r=0 here assumes a scale of order unity
    if (equal_within_roundoff(r[i], 0., r_cutoff, 1.)) {
      get(dpsi_dt)[i] = -2. * profile_->second_deriv(-t);
      get(d2psi_dt2)[i] = 2. * profile_->third_deriv(-t);
      for (size_t d = 0; d < 3; d++) {
        d2psi_dtdx.get(d)[i] = 0.;
      }
    } else {
      const auto dF_out = profile_->first_deriv(r[i] - t);
      const auto dF_in = profile_->first_deriv(-r[i] - t);
      const auto d2F_out = profile_->second_deriv(r[i] - t);
      const auto d2F_in = profile_->second_deriv(-r[i] - t);
      get(dpsi_dt)[i] = (-dF_out + dF_in) / r[i];
      get(d2psi_dt2)[i] = (d2F_out - d2F_in) / r[i];
      const double d2psi_dtdx_isotropic =
          -(d2F_out + d2F_in + get(dpsi_dt)[i]) / square(r[i]);
      for (size_t d = 0; d < 3; d++) {
        d2psi_dtdx.get(d)[i] = d2psi_dtdx_isotropic * x.get(d)[i];
      }
    }
  }
  tuples::TaggedTuple<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<3>>,
                      Tags::dt<ScalarWave::Psi>>
      dt_variables{std::move(d2psi_dt2), std::move(d2psi_dtdx),
                   std::move(dpsi_dt)};
  get<Tags::dt<ScalarWave::Pi>>(dt_variables).get() *= -1.0;
  return dt_variables;
}

void RegularSphericalWave::pup(PUP::er& p) noexcept {
  p | profile_;
}

}  // namespace Solutions
}  // namespace ScalarWave
