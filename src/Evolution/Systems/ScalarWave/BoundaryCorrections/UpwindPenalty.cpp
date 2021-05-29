// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"

#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

namespace ScalarWave::BoundaryCorrections {
template <size_t Dim>
UpwindPenalty<Dim>::UpwindPenalty(CkMigrateMessage* msg) noexcept
    : BoundaryCorrection<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<BoundaryCorrection<Dim>> UpwindPenalty<Dim>::get_clone()
    const noexcept {
  return std::make_unique<UpwindPenalty>(*this);
}

template <size_t Dim>
void UpwindPenalty<Dim>::pup(PUP::er& p) {
  BoundaryCorrection<Dim>::pup(p);
}

template <size_t Dim>
double UpwindPenalty<Dim>::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_psi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_v_zero,
    const gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_plus,
    const gsl::not_null<Scalar<DataVector>*> packaged_char_speed_v_minus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_n_times_v_plus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_n_times_v_minus,
    const gsl::not_null<Scalar<DataVector>*> packaged_char_speed_gamma2_v_psi,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_char_speeds,

    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const Scalar<DataVector>& psi,

    const Scalar<DataVector>& constraint_gamma2,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
    const noexcept {
  if (normal_dot_mesh_velocity.has_value()) {
    get<0>(*packaged_char_speeds) = -get(*normal_dot_mesh_velocity);
    get<1>(*packaged_char_speeds) = 1.0 - get(*normal_dot_mesh_velocity);
    get<2>(*packaged_char_speeds) = -1.0 - get(*normal_dot_mesh_velocity);
  } else {
    get<0>(*packaged_char_speeds) = 0.0;
    get<1>(*packaged_char_speeds) = 1.0;
    get<2>(*packaged_char_speeds) = -1.0;
  }

  // Computes the contribution to the boundary correction from one side of the
  // interface.
  //
  // Note: when UpwindPenalty::dg_boundary_terms() is called, an Element passes
  // in its own packaged data to fill the interior fields, and its neighbor's
  // packaged data to fill the exterior fields. This introduces a sign flip for
  // each normal used in computing the exterior fields.
  get(*packaged_char_speed_gamma2_v_psi) = get(constraint_gamma2) * get(psi);
  {
    // Use v_psi allocation as n^i Phi_i
    dot_product(packaged_char_speed_v_psi, normal_covector, phi);
    const auto& normal_dot_phi = get(*packaged_char_speed_v_psi);

    for (size_t i = 0; i < Dim; ++i) {
      packaged_char_speed_v_zero->get(i) =
          get<0>(*packaged_char_speeds) *
          (phi.get(i) - normal_covector.get(i) * normal_dot_phi);
    }

    get(*packaged_char_speed_v_plus) =
        get<1>(*packaged_char_speeds) *
        (get(pi) + normal_dot_phi - get(*packaged_char_speed_gamma2_v_psi));
    get(*packaged_char_speed_v_minus) =
        get<2>(*packaged_char_speeds) *
        (get(pi) - normal_dot_phi - get(*packaged_char_speed_gamma2_v_psi));
  }

  for (size_t d = 0; d < Dim; ++d) {
    packaged_char_speed_n_times_v_plus->get(d) =
        get(*packaged_char_speed_v_plus) * normal_covector.get(d);
    packaged_char_speed_n_times_v_minus->get(d) =
        get(*packaged_char_speed_v_minus) * normal_covector.get(d);
  }

  get(*packaged_char_speed_v_psi) = get<0>(*packaged_char_speeds) * get(psi);
  get(*packaged_char_speed_gamma2_v_psi) *= get<0>(*packaged_char_speeds);

  return max(max(get<0>(*packaged_char_speeds), get<1>(*packaged_char_speeds),
                 get<2>(*packaged_char_speeds)));
}

template <size_t Dim>
void UpwindPenalty<Dim>::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> pi_boundary_correction,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,
    const gsl::not_null<Scalar<DataVector>*> psi_boundary_correction,

    const Scalar<DataVector>& char_speed_v_psi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_int,
    const Scalar<DataVector>& char_speed_v_plus_int,
    const Scalar<DataVector>& char_speed_v_minus_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_normal_times_v_plus_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_normal_times_v_minus_int,
    const Scalar<DataVector>& char_speed_constraint_gamma2_v_psi_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& char_speeds_int,

    const Scalar<DataVector>& char_speed_v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_ext,
    const Scalar<DataVector>& char_speed_v_plus_ext,
    const Scalar<DataVector>& char_speed_v_minus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_plus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_minus_ext,
    const Scalar<DataVector>& char_speed_constraint_gamma2_v_psi_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& char_speeds_ext,
    dg::Formulation /*dg_formulation*/) const noexcept {
  const size_t num_pts = char_speeds_int[0].size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                       ::Tags::TempScalar<6>, ::Tags::TempScalar<7>>>
      buffer(num_pts);
  DataVector& weighted_lambda_psi_int = get(get<::Tags::TempScalar<0>>(buffer));
  weighted_lambda_psi_int = step_function(-char_speeds_int[0]);
  DataVector& weighted_lambda_psi_ext = get(get<::Tags::TempScalar<1>>(buffer));
  weighted_lambda_psi_ext = -step_function(char_speeds_ext[0]);

  DataVector& weighted_lambda_zero_int =
      get(get<::Tags::TempScalar<2>>(buffer));
  weighted_lambda_zero_int = step_function(-char_speeds_int[0]);
  DataVector& weighted_lambda_zero_ext =
      get(get<::Tags::TempScalar<3>>(buffer));
  weighted_lambda_zero_ext = -step_function(char_speeds_ext[0]);

  DataVector& weighted_lambda_plus_int =
      get(get<::Tags::TempScalar<4>>(buffer));
  weighted_lambda_plus_int = step_function(-char_speeds_int[1]);
  DataVector& weighted_lambda_plus_ext =
      get(get<::Tags::TempScalar<5>>(buffer));
  weighted_lambda_plus_ext = -step_function(char_speeds_ext[1]);

  DataVector& weighted_lambda_minus_int =
      get(get<::Tags::TempScalar<6>>(buffer));
  weighted_lambda_minus_int = step_function(-char_speeds_int[2]);
  DataVector& weighted_lambda_minus_ext =
      get(get<::Tags::TempScalar<7>>(buffer));
  weighted_lambda_minus_ext = -step_function(char_speeds_ext[2]);

  // D_psi = Theta(-lambda_psi^{ext}) lambda_psi^{ext} v_psi^{ext}
  //       - Theta(-lambda_psi^{int}) lambda_psi^{int} v_psi^{int}
  // where the unit normals on both sides point in the same direction, out
  // of the current element. Since lambda_psi from the neighbor is computing
  // with the normal vector pointing into the current element in the code,
  // we need to swap the sign of lambda_psi^{ext}. Theta is the heaviside step
  // function.
  psi_boundary_correction->get() =
      weighted_lambda_psi_ext * get(char_speed_v_psi_ext) -
      weighted_lambda_psi_int * get(char_speed_v_psi_int);

  get(*pi_boundary_correction) =
      0.5 * (weighted_lambda_plus_ext * get(char_speed_v_plus_ext) +
             weighted_lambda_minus_ext * get(char_speed_v_minus_ext)) +
      weighted_lambda_psi_ext * get(char_speed_constraint_gamma2_v_psi_ext)

      - 0.5 * (weighted_lambda_plus_int * get(char_speed_v_plus_int) +
               weighted_lambda_minus_int * get(char_speed_v_minus_int)) -
      weighted_lambda_psi_int * get(char_speed_constraint_gamma2_v_psi_int);

  for (size_t d = 0; d < Dim; ++d) {
    phi_boundary_correction->get(d) =
        0.5 * (weighted_lambda_plus_ext *
                   char_speed_minus_normal_times_v_plus_ext.get(d) -
               weighted_lambda_minus_ext *
                   char_speed_minus_normal_times_v_minus_ext.get(d)) +
        weighted_lambda_zero_ext * char_speed_v_zero_ext.get(d)

        - 0.5 * (weighted_lambda_plus_int *
                     char_speed_normal_times_v_plus_int.get(d) -
                 weighted_lambda_minus_int *
                     char_speed_normal_times_v_minus_int.get(d)) -
        weighted_lambda_zero_int * char_speed_v_zero_int.get(d);
  }
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID UpwindPenalty<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class UpwindPenalty<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarWave::BoundaryCorrections
