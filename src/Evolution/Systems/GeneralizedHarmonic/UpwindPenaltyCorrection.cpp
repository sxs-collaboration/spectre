// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/UpwindPenaltyCorrection.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace GeneralizedHarmonic {
template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::package_data(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_v_spacetime_metric,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_v_zero,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_v_plus,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_v_minus,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_n_times_v_plus,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_n_times_v_minus,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_char_speed_gamma2_v_spacetime_metric,
    const gsl::not_null<tnsr::a<DataVector, 3, Frame::Inertial>*>
        packaged_char_speeds,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& v_zero,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_plus,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& v_minus,
    const std::array<DataVector, 4>& char_speeds,
    const Scalar<DataVector>& constraint_gamma2,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  for (size_t storage_index = 0; storage_index < v_spacetime_metric.size();
       ++storage_index) {
    packaged_char_speed_v_spacetime_metric->operator[](storage_index) =
        char_speeds[0] * v_spacetime_metric[storage_index];
    packaged_char_speed_gamma2_v_spacetime_metric->operator[](storage_index) =
        get(constraint_gamma2) *
        packaged_char_speed_v_spacetime_metric->operator[](storage_index);
  }

  *packaged_char_speed_v_zero = v_zero;
  for (size_t storage_index = 0; storage_index < v_zero.size();
       ++storage_index) {
    packaged_char_speed_v_zero->operator[](storage_index) *= char_speeds[1];
  }

  for (size_t storage_index = 0; storage_index < v_plus.size();
       ++storage_index) {
    packaged_char_speed_v_plus->operator[](storage_index) =
        char_speeds[2] * v_plus[storage_index];
    packaged_char_speed_v_minus->operator[](storage_index) =
        char_speeds[3] * v_minus[storage_index];
  }

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = 0; b <= a; ++b) {
      for (size_t i = 0; i < Dim; ++i) {
        packaged_char_speed_n_times_v_plus->get(i, a, b) =
            packaged_char_speed_v_plus->get(a, b) *
            interface_unit_normal.get(i);
        packaged_char_speed_n_times_v_minus->get(i, a, b) =
            packaged_char_speed_v_minus->get(a, b) *
            interface_unit_normal.get(i);
      }
    }
  }
  for (size_t i = 0; i < 4; ++i) {
    (*packaged_char_speeds)[i] = gsl::at(char_speeds, i);
  }
}

template <size_t Dim>
void UpwindPenaltyCorrection<Dim>::operator()(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        spacetime_metric_boundary_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        pi_boundary_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        phi_boundary_correction,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        char_speed_v_spacetime_metric_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_plus_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_minus_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
        char_speed_normal_times_v_plus_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
        char_speed_normal_times_v_minus_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        char_speed_constraint_gamma2_v_spacetime_metric_int,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_int,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        char_speed_v_spacetime_metric_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& char_speed_v_zero_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_plus_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& char_speed_v_minus_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_plus_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
        char_speed_minus_normal_times_v_minus_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        char_speed_constraint_gamma2_v_spacetime_metric_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext)
    const noexcept {
  const size_t num_pts = char_speeds_int[0].size();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                       ::Tags::TempScalar<6>, ::Tags::TempScalar<7>>>
      buffer(num_pts);
  DataVector& weighted_lambda_spacetime_metric_int =
      get(get<::Tags::TempScalar<0>>(buffer));
  weighted_lambda_spacetime_metric_int = step_function(-char_speeds_int[0]);
  DataVector& weighted_lambda_spacetime_metric_ext =
      get(get<::Tags::TempScalar<1>>(buffer));
  weighted_lambda_spacetime_metric_ext = -step_function(char_speeds_ext[0]);

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

  for (size_t storage_index = 0;
       storage_index < char_speed_v_spacetime_metric_ext.size();
       ++storage_index) {
    spacetime_metric_boundary_correction->operator[](storage_index) =
        weighted_lambda_spacetime_metric_ext *
            char_speed_v_spacetime_metric_ext[storage_index] -
        weighted_lambda_spacetime_metric_int *
            char_speed_v_spacetime_metric_int[storage_index];
  }

  for (size_t storage_index = 0;
       storage_index < char_speed_v_spacetime_metric_ext.size();
       ++storage_index) {
    (*pi_boundary_correction)[storage_index] =
        0.5 * (weighted_lambda_plus_ext * char_speed_v_plus_ext[storage_index] +
               weighted_lambda_minus_ext *
                   char_speed_v_minus_ext[storage_index]) +
        weighted_lambda_spacetime_metric_ext *
            char_speed_constraint_gamma2_v_spacetime_metric_ext[storage_index]

        -
        0.5 * (weighted_lambda_plus_int * char_speed_v_plus_int[storage_index] +
               weighted_lambda_minus_int *
                   char_speed_v_minus_int[storage_index]) -
        weighted_lambda_spacetime_metric_int *
            char_speed_constraint_gamma2_v_spacetime_metric_int[storage_index];
  }

  for (size_t storage_index = 0;
       storage_index < phi_boundary_correction->size(); ++storage_index) {
    // Overall minus sign on ext because of normal vector is opposite direction.
    (*phi_boundary_correction)[storage_index] =
        0.5 * (weighted_lambda_plus_ext *
                   char_speed_minus_normal_times_v_plus_ext[storage_index] -
               weighted_lambda_minus_ext *
                   char_speed_minus_normal_times_v_minus_ext[storage_index]) +
        weighted_lambda_zero_ext * char_speed_v_zero_ext[storage_index]

        - 0.5 * (weighted_lambda_plus_int *
                     char_speed_normal_times_v_plus_int[storage_index] -
                 weighted_lambda_minus_int *
                     char_speed_normal_times_v_minus_int[storage_index]) -
        weighted_lambda_zero_int * char_speed_v_zero_int[storage_index];
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template class UpwindPenaltyCorrection<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace GeneralizedHarmonic
/// \endcond
