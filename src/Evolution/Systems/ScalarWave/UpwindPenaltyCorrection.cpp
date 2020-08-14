// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/UpwindPenaltyCorrection.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"   // IWYU pragma: keep
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

/// \cond
namespace ScalarWave {
template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::package_data(
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
    const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
        packaged_char_speeds,

    const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const std::array<DataVector, 4>& char_speeds,
    const Scalar<DataVector>& constraint_gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Computes the contribution to the numerical flux from one side of the
  // interface.
  //
  // Note: when PenaltyFlux::operator() is called, an Element passes in its own
  // packaged data to fill the interior fields, and its neighbor's packaged data
  // to fill the exterior fields. This introduces a sign flip for each normal
  // used in computing the exterior fields.
  get(*packaged_char_speed_v_psi) = char_speeds[0] * get(v_psi);
  *packaged_char_speed_v_zero = v_zero;
  for (size_t i = 0; i < Dim; ++i) {
    packaged_char_speed_v_zero->get(i) *= char_speeds[1];
  }
  get(*packaged_char_speed_v_plus) = char_speeds[2] * get(v_plus);
  get(*packaged_char_speed_v_minus) = char_speeds[3] * get(v_minus);
  for (size_t d = 0; d < Dim; ++d) {
    packaged_char_speed_n_times_v_plus->get(d) =
        get(*packaged_char_speed_v_plus) * interface_unit_normal.get(d);
    packaged_char_speed_n_times_v_minus->get(d) =
        get(*packaged_char_speed_v_minus) * interface_unit_normal.get(d);
  }
  for (size_t i = 0; i < 4; ++i) {
    (*packaged_char_speeds)[i] = gsl::at(char_speeds, i);
  }
  get(*packaged_char_speed_gamma2_v_psi) =
      get(constraint_gamma2) * get(*packaged_char_speed_v_psi);
}

template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::operator()(
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
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

    const Scalar<DataVector>& char_speed_v_psi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_ext,
    const Scalar<DataVector>& char_speed_v_plus_ext,
    const Scalar<DataVector>& char_speed_v_minus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_plus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_minus_ext,
    const Scalar<DataVector>& char_speed_constraint_gamma2_v_psi_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext)
    const noexcept {
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
  weighted_lambda_zero_int = step_function(-char_speeds_int[1]);
  DataVector& weighted_lambda_zero_ext =
      get(get<::Tags::TempScalar<3>>(buffer));
  weighted_lambda_zero_ext = -step_function(char_speeds_ext[1]);

  DataVector& weighted_lambda_plus_int =
      get(get<::Tags::TempScalar<4>>(buffer));
  weighted_lambda_plus_int = step_function(-char_speeds_int[2]);
  DataVector& weighted_lambda_plus_ext =
      get(get<::Tags::TempScalar<5>>(buffer));
  weighted_lambda_plus_ext = -step_function(char_speeds_ext[2]);

  DataVector& weighted_lambda_minus_int =
      get(get<::Tags::TempScalar<6>>(buffer));
  weighted_lambda_minus_int = step_function(-char_speeds_int[3]);
  DataVector& weighted_lambda_minus_ext =
      get(get<::Tags::TempScalar<7>>(buffer));
  weighted_lambda_minus_ext = -step_function(char_speeds_ext[3]);

  // D_psi = Theta(-lambda_psi^{ext}) lambda_psi^{ext} v_psi^{ext}
  //       - Theta(-lambda_psi^{int}) lambda_psi^{int} v_psi^{int}
  // where the unit normals on both sides point in the same direction, out
  // of the current element. Since lambda_psi from the neighbor is computing
  // with the normal vector pointing into the current element in the code,
  // we need to swap the sign of lambda_psi^{ext}. Theta is the heaviside step
  // function with Theta(0) = 0.
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
    // Overall minus sign on ext because of normal vector is opposite direction.
    phi_boundary_correction->get(d) =
        -0.5 * (weighted_lambda_minus_ext *
                    char_speed_minus_normal_times_v_minus_ext.get(d) -
                weighted_lambda_plus_ext *
                    char_speed_minus_normal_times_v_plus_ext.get(d)) +
        weighted_lambda_zero_ext * char_speed_v_zero_ext.get(d)

        - 0.5 * (weighted_lambda_plus_int *
                     char_speed_normal_times_v_plus_int.get(d) -
                 weighted_lambda_minus_int *
                     char_speed_normal_times_v_minus_int.get(d)) -
        weighted_lambda_zero_int * char_speed_v_zero_int.get(d);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template class UpwindPenaltyCorrection<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace ScalarWave
/// \endcond
