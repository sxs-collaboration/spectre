// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/SuperposedInverser.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {
template <size_t Dim>
SuperposedInverser<Dim>::SuperposedInverser(double amplitude_a,
                                            double amplitude_b,
                                            std::array<double, Dim> location_a,
                                            std::array<double, Dim> location_b)
    : amplitude_a_(amplitude_a),
      amplitude_b_(amplitude_b),
      location_a_(location_a),
      location_b_(location_b) {}

template <size_t Dim>
std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
SuperposedInverser<Dim>::get_clone() const {
  return std::make_unique<SuperposedInverser>(*this);
}

template <size_t Dim>
void SuperposedInverser<Dim>::pup(PUP::er& p) {
  ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>::pup(p);
  p | amplitude_a_;
  p | amplitude_b_;
  p | location_a_;
  p | location_b_;
}

template <size_t Dim>
tuples::TaggedTuple<::CurvedScalarWave::Tags::Psi>
SuperposedInverser<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::CurvedScalarWave::Tags::Psi> /*meta*/) const {
  Scalar<DataVector> distance_from_loc_a =
      make_with_value<Scalar<DataVector>>(x, 0.);
  Scalar<DataVector> distance_from_loc_b =
      make_with_value<Scalar<DataVector>>(x, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    distance_from_loc_a.get() += square(x.get(i) - location_a_.at(i));
    distance_from_loc_b.get() += square(x.get(i) - location_b_.at(i));
  }
  get(distance_from_loc_a) = sqrt(get(distance_from_loc_a));
  get(distance_from_loc_b) = sqrt(get(distance_from_loc_b));
  Scalar<DataVector> result =
      tenex::evaluate(amplitude_a_ / distance_from_loc_a() +
                      amplitude_b_ / distance_from_loc_b());
  return result;
}

template <size_t Dim>
tuples::TaggedTuple<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>
SuperposedInverser<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x,
    tmpl::list<::CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>> /*meta*/)
    const {
  tnsr::I<DataVector, Dim, Frame::Inertial> displacement_from_loc_a{};
  tnsr::I<DataVector, Dim, Frame::Inertial> displacement_from_loc_b{};
  set_number_of_grid_points(make_not_null(&displacement_from_loc_a), x);
  set_number_of_grid_points(make_not_null(&displacement_from_loc_b), x);

  for (size_t i = 0; i < Dim; ++i) {
    displacement_from_loc_a.get(i) = x.get(i) - location_a_.at(i);
    displacement_from_loc_b.get(i) = x.get(i) - location_b_.at(i);
  }
  const Scalar<DataVector> distance_from_loc_a =
      magnitude(displacement_from_loc_a);
  const Scalar<DataVector> distance_from_loc_b =
      magnitude(displacement_from_loc_b);

  tnsr::i<DataVector, Dim, Frame::Inertial> result{};
  set_number_of_grid_points(make_not_null(&result), x);
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = -amplitude_a_ * displacement_from_loc_a.get(i) /
                        cube(distance_from_loc_a.get()) -
                    amplitude_b_ * displacement_from_loc_b.get(i) /
                        cube(distance_from_loc_b.get());
  }
  return result;
}

template <size_t Dim>
PUP::able::PUP_ID SuperposedInverser<Dim>::my_PUP_ID = 0;  // NOLINT

template <size_t Dim>
bool operator==(const SuperposedInverser<Dim>& lhs,
                const SuperposedInverser<Dim>& rhs) {
  return (lhs.amplitude_a_ == rhs.amplitude_a_ and
          lhs.amplitude_b_ == rhs.amplitude_b_ and
          lhs.location_a_ == rhs.location_a_ and
          lhs.location_b_ == rhs.location_b_);
}

template <size_t Dim>
bool operator!=(const SuperposedInverser<Dim>& lhs,
                const SuperposedInverser<Dim>& rhs) {
  return not(lhs == rhs);
}
}  // namespace ScalarTensor::AnalyticData::ScalarField

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template class ScalarTensor::AnalyticData::ScalarField::SuperposedInverser< \
      DIM(data)>;                                                             \
  template bool ScalarTensor::AnalyticData::ScalarField::operator==(          \
      const ScalarTensor::AnalyticData::ScalarField::SuperposedInverser<DIM(  \
          data)>& lhs,                                                        \
      const ScalarTensor::AnalyticData::ScalarField::SuperposedInverser<DIM(  \
          data)>& rhs);                                                       \
  template bool ScalarTensor::AnalyticData::ScalarField::operator!=(          \
      const ScalarTensor::AnalyticData::ScalarField::SuperposedInverser<DIM(  \
          data)>& lhs,                                                        \
      const ScalarTensor::AnalyticData::ScalarField::SuperposedInverser<DIM(  \
          data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATE, (3))

#undef DIM
#undef INSTANTIATE
