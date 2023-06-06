// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ForceFree/AlfvenWave.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ForceFree::Solutions {

AlfvenWave::AlfvenWave(const double wave_speed, const Options::Context& context)
    : wave_speed_(wave_speed) {
  if (abs(wave_speed_) >= 1.0) {
    PARSE_ERROR(context,
                "The wave speed ("
                    << wave_speed_
                    << ") must be bigger than -1.0 and smaller than 1.0");
  }
  lorentz_factor_ = 1.0 / sqrt(1.0 - square(wave_speed_));
}

AlfvenWave::AlfvenWave(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> AlfvenWave::get_clone()
    const {
  return std::make_unique<AlfvenWave>(*this);
}

void AlfvenWave::pup(PUP::er& p) {
  InitialData::pup(p);
  p | wave_speed_;
  p | lorentz_factor_;
  p | background_spacetime_;
}

PUP::able::PUP_ID AlfvenWave::my_PUP_ID = 0;

DataVector AlfvenWave::wave_profile(const DataVector& x_prime) {
  // Compute the B_z'(=-E_x') at the rest frame of the wave
  auto result = make_with_value<DataVector>(x_prime, 1.0);
  for (size_t i = 0; i < result.size(); ++i) {
    const double& xi = x_prime[i];
    if (xi > -0.1) {
      if (xi <= 0.1) {
        result[i] = 1.15 + 0.15 * sin(5 * M_PI * xi);
      } else {
        result[i] = 1.3;
      }
    }
  }
  return result;
}

DataVector AlfvenWave::charge_density(const DataVector& x_prime) {
  // Compute the charge density q = Div(E) at the rest frame of the wave
  auto result = make_with_value<DataVector>(x_prime, 0.0);
  for (size_t i = 0; i < result.size(); ++i) {
    const double& xi = x_prime[i];
    if ((xi > -0.1) and (xi <= 0.1)) {
      result[i] = -0.75 * M_PI * cos(5.0 * M_PI * xi);
    }
  }
  return result;
}

tuples::TaggedTuple<Tags::TildeE> AlfvenWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    tmpl::list<Tags::TildeE> /*meta*/) const {
  const auto& x_coord = get<0>(x);
  auto electric_field = make_with_value<tnsr::I<DataVector, 3>>(x, 0.0);

  // E_x
  get<0>(electric_field) =
      -wave_profile(lorentz_factor_ * (x_coord - wave_speed_ * t));
  // E_y
  get<1>(electric_field) =
      -lorentz_factor_ * wave_speed_ * get<0>(electric_field);
  // E_z
  get<2>(electric_field) = lorentz_factor_ * (1.0 - wave_speed_);

  return electric_field;
}

tuples::TaggedTuple<Tags::TildeB> AlfvenWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    tmpl::list<Tags::TildeB> /*meta*/) const {
  const auto& x_coord = get<0>(x);
  auto magnetic_field = make_with_value<tnsr::I<DataVector, 3>>(x, 1.0);
  // note) B_x = 1.0
  // B_y
  get<1>(magnetic_field) = lorentz_factor_ * (1.0 - wave_speed_);
  // B_z
  get<2>(magnetic_field) =
      lorentz_factor_ *
      wave_profile(lorentz_factor_ * (x_coord - wave_speed_ * t));

  return magnetic_field;
}

tuples::TaggedTuple<Tags::TildePsi> AlfvenWave::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> AlfvenWave::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> AlfvenWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    tmpl::list<Tags::TildeQ> /*meta*/) const {
  return {Scalar<DataVector>{
      lorentz_factor_ *
      charge_density(lorentz_factor_ * (get<0>(x) - wave_speed_ * t))}};
}

bool operator==(const AlfvenWave& lhs, const AlfvenWave& rhs) {
  return lhs.wave_speed_ == rhs.wave_speed_ and
         lhs.lorentz_factor_ == rhs.lorentz_factor_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const AlfvenWave& lhs, const AlfvenWave& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::Solutions
