// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/AnalyticConstant.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Characteristics.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeString.hpp"

namespace CurvedScalarWave::BoundaryConditions {

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
AnalyticConstant<Dim>::get_clone() const {
  return std::make_unique<AnalyticConstant>(*this);
}

template <size_t Dim>
void AnalyticConstant<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | amplitude_;
}

template <size_t Dim>
AnalyticConstant<Dim>::AnalyticConstant(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
AnalyticConstant<Dim>::AnalyticConstant(const double amplitude)
    : amplitude_(amplitude) {}

template <size_t Dim>
std::optional<std::string> AnalyticConstant<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> psi,
    const gsl::not_null<Scalar<DataVector>*> pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift,
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inverse_spatial_metric,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim>& /*normal_covector*/,
    const tnsr::I<DataVector, Dim>& /*normal_vector*/,
    const tnsr::II<DataVector, Dim, Frame::Inertial>&
        inverse_spatial_metric_interior,
    const Scalar<DataVector>& gamma1_interior,
    const Scalar<DataVector>& gamma2_interior,
    const Scalar<DataVector>& lapse_interior,
    const tnsr::I<DataVector, Dim>& shift_interior) const {
  *psi = make_with_value<Scalar<DataVector>>(gamma1_interior, amplitude_);
  *pi = make_with_value<Scalar<DataVector>>(gamma1_interior, 0.0);
  *phi = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      gamma1_interior, 0.0);
  *lapse = lapse_interior;
  *shift = shift_interior;
  *inverse_spatial_metric = inverse_spatial_metric_interior;
  *gamma1 = gamma1_interior;
  *gamma2 = gamma2_interior;

  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID AnalyticConstant<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class AnalyticConstant<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace CurvedScalarWave::BoundaryConditions
