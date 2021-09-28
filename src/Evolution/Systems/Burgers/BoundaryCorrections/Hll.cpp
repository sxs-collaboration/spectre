// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/BoundaryCorrections/Hll.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Utilities/Gsl.hpp"

namespace Burgers::BoundaryCorrections {
Hll::Hll(CkMigrateMessage* msg) : BoundaryCorrection(msg) {}

std::unique_ptr<BoundaryCorrection> Hll::get_clone() const {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) { BoundaryCorrection::pup(p); }

double Hll::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_u,
    const gsl::not_null<Scalar<DataVector>*> packaged_char_speed,
    const Scalar<DataVector>& u,
    const tnsr::I<DataVector, 1, Frame::Inertial>& flux_u,
    const tnsr::i<DataVector, 1, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, 1, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity) {
  get(*packaged_char_speed) = sign(get<0>(normal_covector)) * get(u);
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_char_speed) -= get(*normal_dot_mesh_velocity);
  }
  *packaged_u = u;
  normal_dot_flux(packaged_normal_dot_flux_u, normal_covector, flux_u);
  return max(abs(get(*packaged_char_speed)));
}

void Hll::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_u,
    const Scalar<DataVector>& u_int,
    const Scalar<DataVector>& normal_dot_flux_u_int,
    const Scalar<DataVector>& char_speed_int, const Scalar<DataVector>& u_ext,
    const Scalar<DataVector>& normal_dot_flux_u_ext,
    const Scalar<DataVector>& char_speed_ext,
    const dg::Formulation dg_formulation) {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> temps{
      get(u_int).size()};
  get(get<::Tags::TempScalar<0>>(temps)) =
      min(0.0, get(char_speed_int), -get(char_speed_ext));
  get(get<::Tags::TempScalar<1>>(temps)) =
      max(0.0, get(char_speed_int), -get(char_speed_ext));
  const DataVector& lambda_min = get(get<::Tags::TempScalar<0>>(temps));
  const DataVector& lambda_max = get(get<::Tags::TempScalar<1>>(temps));

  get(*boundary_correction_u) =
      ((lambda_max * get(normal_dot_flux_u_int) +
        lambda_min * get(normal_dot_flux_u_ext)) +
       lambda_max * lambda_min * (get(u_ext) - get(u_int))) /
      (lambda_max - lambda_min);
  if (dg_formulation == dg::Formulation::StrongInertial) {
    get(*boundary_correction_u) -= get(normal_dot_flux_u_int);
  }
}
}  // namespace Burgers::BoundaryCorrections

// NOLINTNEXTLINE
PUP::able::PUP_ID Burgers::BoundaryCorrections::Hll::my_PUP_ID = 0;
