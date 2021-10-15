// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace GeneralizedHarmonic::BoundaryConditions {
template <size_t Dim>
DirichletMinkowski<Dim>::DirichletMinkowski(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletMinkowski<Dim>::get_clone() const {
  return std::make_unique<DirichletMinkowski>(*this);
}

template <size_t Dim>
void DirichletMinkowski<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
std::optional<std::string> DirichletMinkowski<Dim>::dg_ghost(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
        inv_spatial_metric,
    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
    const Scalar<DataVector>& interior_gamma1,
    const Scalar<DataVector>& interior_gamma2) const {
  *gamma1 = interior_gamma1;
  *gamma2 = interior_gamma2;

  // Hard-code Minkowski values to avoid passing coords or time as arguments
  *spacetime_metric =
      make_with_value<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
          interior_gamma1, 0.0);
  get<0, 0>(*spacetime_metric) = -1.0;
  for (size_t i = 1; i < Dim + 1; ++i) {
    (*spacetime_metric).get(i, i) = 1.0;
  }
  *pi = make_with_value<tnsr::aa<DataVector, Dim, Frame::Inertial>>(
      interior_gamma1, 0.0);
  *phi = make_with_value<tnsr::iaa<DataVector, Dim, Frame::Inertial>>(
      interior_gamma1, 0.0);

  *lapse = make_with_value<Scalar<DataVector>>(interior_gamma1, 1.0);
  *shift = make_with_value<tnsr::I<DataVector, Dim, Frame::Inertial>>(
      interior_gamma1, 0.0);
  *inv_spatial_metric =
      make_with_value<tnsr::II<DataVector, Dim, Frame::Inertial>>(
          interior_gamma1, 0.0);
  for (size_t i = 0; i < Dim; ++i) {
    (*inv_spatial_metric).get(i, i) = 1.0;
  }

  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletMinkowski<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class DirichletMinkowski<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic::BoundaryConditions
