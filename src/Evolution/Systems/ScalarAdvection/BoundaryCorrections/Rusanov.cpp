// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/BoundaryCorrections/Rusanov.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarAdvection::BoundaryCorrections {
template <size_t Dim>
Rusanov<Dim>::Rusanov(CkMigrateMessage* msg) noexcept
    : BoundaryCorrection<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<BoundaryCorrection<Dim>> Rusanov<Dim>::get_clone()
    const noexcept {
  return std::make_unique<Rusanov>(*this);
}

template <size_t Dim>
void Rusanov<Dim>::pup(PUP::er& p) {
  BoundaryCorrection<Dim>::pup(p);
}

template <size_t Dim>
double Rusanov<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_abs_char_speed,
    const Scalar<DataVector>& u,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& flux_u,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity_field,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>&
        normal_dot_mesh_velocity) noexcept {
  Scalar<DataVector>& normal_dot_velocity = *packaged_u;
  dot_product(make_not_null(&normal_dot_velocity), velocity_field,
              normal_covector);
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_abs_char_speed) =
        abs(get(normal_dot_velocity) - get(*normal_dot_mesh_velocity));
  } else {
    get(*packaged_abs_char_speed) = abs(get(normal_dot_velocity));
  }
  *packaged_u = u;
  normal_dot_flux(packaged_normal_dot_flux_u, normal_covector, flux_u);
  return max(get(*packaged_abs_char_speed));
}

template <size_t Dim>
void Rusanov<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_u,
    const Scalar<DataVector>& u_int,
    const Scalar<DataVector>& normal_dot_flux_u_int,
    const Scalar<DataVector>& abs_char_speed_int,
    const Scalar<DataVector>& u_ext,
    const Scalar<DataVector>& normal_dot_flux_u_ext,
    const Scalar<DataVector>& abs_char_speed_ext,
    const dg::Formulation dg_formulation) noexcept {
  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_u) =
        0.5 * (get(normal_dot_flux_u_int) - get(normal_dot_flux_u_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(u_ext) - get(u_int));
  } else {
    get(*boundary_correction_u) =
        -0.5 * (get(normal_dot_flux_u_int) + get(normal_dot_flux_u_ext)) -
        0.5 * max(get(abs_char_speed_int), get(abs_char_speed_ext)) *
            (get(u_ext) - get(u_int));
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID
    ScalarAdvection::BoundaryCorrections::Rusanov<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Rusanov<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarAdvection::BoundaryCorrections
