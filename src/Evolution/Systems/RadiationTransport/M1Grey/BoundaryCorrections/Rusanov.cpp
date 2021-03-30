// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/BoundaryCorrections/Rusanov.hpp"

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

/// \cond
namespace RadiationTransport::M1Grey::BoundaryCorrections::Rusanov_detail {

void dg_package_data_impl(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_e,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_e,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_e,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<
        tnsr::I<DataVector, 3, Frame::Inertial>>& /*mesh_velocity*/,
    const std::optional<
        Scalar<DataVector>>& /*normal_dot_mesh_velocity*/) noexcept {
  *packaged_tilde_e = tilde_e;
  *packaged_tilde_s = tilde_s;

  dot_product(packaged_normal_dot_flux_tilde_e, flux_tilde_e, normal_covector);
  for (size_t i = 0; i < 3; ++i) {
    packaged_normal_dot_flux_tilde_s->get(i) =
        get<0>(normal_covector) * flux_tilde_s.get(0, i);
    for (size_t j = 1; j < 3; ++j) {
      packaged_normal_dot_flux_tilde_s->get(i) +=
          normal_covector.get(j) * flux_tilde_s.get(j, i);
    }
  }
}

void dg_boundary_terms_impl(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_e,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const Scalar<DataVector>& tilde_e_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_e_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const Scalar<DataVector>& tilde_e_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_e_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    dg::Formulation dg_formulation) noexcept {
  constexpr double max_abs_char_speed = 1.0;
  if (dg_formulation == dg::Formulation::WeakInertial) {
    get(*boundary_correction_tilde_e) =
        0.5 * (get(normal_dot_flux_tilde_e_int) -
               get(normal_dot_flux_tilde_e_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_e_ext) - get(tilde_e_int));
    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          0.5 * (normal_dot_flux_tilde_s_int.get(i) -
                 normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_s_ext.get(i) - tilde_s_int.get(i));
    }
  } else {
    get(*boundary_correction_tilde_e) =
        -0.5 * (get(normal_dot_flux_tilde_e_int) +
                get(normal_dot_flux_tilde_e_ext)) -
        0.5 * max_abs_char_speed * (get(tilde_e_ext) - get(tilde_e_int));
    for (size_t i = 0; i < 3; ++i) {
      boundary_correction_tilde_s->get(i) =
          -0.5 * (normal_dot_flux_tilde_s_int.get(i) +
                  normal_dot_flux_tilde_s_ext.get(i)) -
          0.5 * max_abs_char_speed * (tilde_s_ext.get(i) - tilde_s_int.get(i));
    }
  }
}

}  // namespace RadiationTransport::M1Grey::BoundaryCorrections::Rusanov_detail
/// \endcond
