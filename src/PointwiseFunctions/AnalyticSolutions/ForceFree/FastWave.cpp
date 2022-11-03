// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ForceFree/FastWave.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ForceFree::Solutions {

FastWave::FastWave(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> FastWave::get_clone()
    const {
  return std::make_unique<FastWave>(*this);
}

void FastWave::pup(PUP::er& p) {
  InitialData::pup(p);
  p | background_spacetime_;
}

PUP::able::PUP_ID FastWave::my_PUP_ID = 0;

DataVector FastWave::initial_profile(const DataVector& coords) {
  // Compute the initial functional form of B_y(=E_z)
  auto result = make_with_value<DataVector>(coords, 1.0);
  for (size_t i = 0; i < result.size(); ++i) {
    const double& x = coords[i];
    if (x > -0.1) {
      if (x <= 0.1) {
        result[i] = -1.5 * x + 0.85;
      } else {
        result[i] = 0.7;
      }
    }
  }
  return result;
}

tuples::TaggedTuple<Tags::TildeE> FastWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    tmpl::list<Tags::TildeE> /*meta*/) {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(x, 0.0);
  // E_z = -B_y
  result.get(2) = -initial_profile(x.get(0) - t);
  return result;
}

tuples::TaggedTuple<Tags::TildeB> FastWave::variables(
    const tnsr::I<DataVector, 3>& x, double t,
    tmpl::list<Tags::TildeB> /*meta*/) {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(x, 0.0);
  result.get(0) = 1.0;  // B_x = 1.0
  result.get(1) = initial_profile(x.get(0) - t);
  return result;
}

tuples::TaggedTuple<Tags::TildePsi> FastWave::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> FastWave::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> FastWave::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

bool operator==(const FastWave& lhs, const FastWave& rhs) {
  return lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const FastWave& lhs, const FastWave& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::Solutions
