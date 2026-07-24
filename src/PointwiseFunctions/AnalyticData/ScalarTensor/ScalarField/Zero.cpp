// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Zero.hpp"

#include <memory>

#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {
template <size_t Dim>
std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
Zero<Dim>::get_clone() const {
  return std::make_unique<Zero>(*this);
}

template <size_t Dim>
PUP::able::PUP_ID Zero<Dim>::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
bool operator==(const Zero<Dim>& /*lhs*/, const Zero<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const Zero<Dim>& lhs, const Zero<Dim>& rhs) {
  return not(lhs == rhs);
}
}  // namespace ScalarTensor::AnalyticData::ScalarField

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template class ScalarTensor::AnalyticData::ScalarField::Zero<DIM(data)>;  \
  template bool ScalarTensor::AnalyticData::ScalarField::operator==(        \
      const ScalarTensor::AnalyticData::ScalarField::Zero<DIM(data)>& lhs,  \
      const ScalarTensor::AnalyticData::ScalarField::Zero<DIM(data)>& rhs); \
  template bool ScalarTensor::AnalyticData::ScalarField::operator!=(        \
      const ScalarTensor::AnalyticData::ScalarField::Zero<DIM(data)>& lhs,  \
      const ScalarTensor::AnalyticData::ScalarField::Zero<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef DIM
#undef INSTANTIATE
