// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ForceFree/FfeBreakdown.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ForceFree::AnalyticData {

FfeBreakdown::FfeBreakdown(CkMigrateMessage* msg) : InitialData(msg) {}

std::unique_ptr<evolution::initial_data::InitialData> FfeBreakdown::get_clone()
    const {
  return std::make_unique<FfeBreakdown>(*this);
}

void FfeBreakdown::pup(PUP::er& p) {
  InitialData::pup(p);
  p | background_spacetime_;
}

PUP::able::PUP_ID FfeBreakdown::my_PUP_ID = 0;

tuples::TaggedTuple<Tags::TildeE> FfeBreakdown::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeE> /*meta*/) {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 0.0);
  result.get(1) = 0.5;
  result.get(2) = -0.5;
  return result;
}

tuples::TaggedTuple<Tags::TildeB> FfeBreakdown::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeB> /*meta*/) {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(coords, 1.0);
  for (size_t i = 0; i < result.get(0).size(); ++i) {
    const double& x = coords.get(0)[i];
    if (x > -0.1) {
      if (x <= 0.1) {
        result.get(1)[i] = -10.0 * x;
        result.get(2)[i] = -10.0 * x;
      } else {
        result.get(1)[i] = -1.0;
        result.get(2)[i] = -1.0;
      }
    }
  }
  return result;
}

tuples::TaggedTuple<Tags::TildePsi> FfeBreakdown::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePsi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildePhi> FfeBreakdown::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildePhi> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

tuples::TaggedTuple<Tags::TildeQ> FfeBreakdown::variables(
    const tnsr::I<DataVector, 3>& coords, tmpl::list<Tags::TildeQ> /*meta*/) {
  return {make_with_value<Scalar<DataVector>>(coords, 0.0)};
}

bool operator==(const FfeBreakdown& lhs, const FfeBreakdown& rhs) {
  return lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const FfeBreakdown& lhs, const FfeBreakdown& rhs) {
  return not(lhs == rhs);
}

}  // namespace ForceFree::AnalyticData
