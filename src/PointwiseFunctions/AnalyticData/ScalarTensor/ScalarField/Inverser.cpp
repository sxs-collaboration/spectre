// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/Inverser.hpp"

#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {
template <size_t Dim>
Inverser<Dim>::Inverser(double amplitude) : amplitude_(amplitude) {}

template <size_t Dim>
std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
Inverser<Dim>::get_clone() const {
  return std::make_unique<Inverser>(*this);
}

template <size_t Dim>
void Inverser<Dim>::pup(PUP::er& p) {
  ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>::pup(p);
  p | amplitude_;
}

template <size_t Dim>
tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi> Inverser<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
  const Scalar<DataVector> distance_from_center = magnitude(x);
  Scalar<DataVector> result =
      tenex::evaluate(amplitude_ / distance_from_center());
  return result;
}

template <size_t Dim>
tuples::TaggedTuple<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>
Inverser<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>> /*meta*/)
    const {
  const DataVector distance_from_center = get(magnitude(x));

  tnsr::i<DataVector, Dim, Frame::Inertial> result{};
  set_number_of_grid_points(make_not_null(&result), x);

  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = -amplitude_ * x.get(i) / cube(distance_from_center);
  }

  return result;
}

template <size_t Dim>
PUP::able::PUP_ID Inverser<Dim>::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
bool operator==(const Inverser<Dim>& lhs, const Inverser<Dim>& rhs) {
  return (lhs.amplitude_ == rhs.amplitude_);
}

template <size_t Dim>
bool operator!=(const Inverser<Dim>& lhs, const Inverser<Dim>& rhs) {
  return not(lhs == rhs);
}

}  // namespace ScalarTensor::AnalyticData::ScalarField

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class ScalarTensor::AnalyticData::ScalarField::Inverser<DIM(data)>; \
  template bool ScalarTensor::AnalyticData::ScalarField::operator==(           \
      const ScalarTensor::AnalyticData::ScalarField::Inverser<DIM(data)>& lhs, \
      const ScalarTensor::AnalyticData::ScalarField::Inverser<DIM(data)>&      \
          rhs);                                                                \
  template bool ScalarTensor::AnalyticData::ScalarField::operator!=(           \
      const ScalarTensor::AnalyticData::ScalarField::Inverser<DIM(data)>& lhs, \
      const ScalarTensor::AnalyticData::ScalarField::Inverser<DIM(data)>&      \
          rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef DIM
#undef INSTANTIATE
