// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"

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

namespace GeneralizedHarmonic::BoundaryCorrections {
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

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,

    const Scalar<DataVector>& constraint_gamma1,
    const Scalar<DataVector>& constraint_gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_vector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity)
    const noexcept {
  // Compute the char speeds without the mesh movement, then add the mesh
  // movement. We compute the zero-speed first since it is just the normal
  // dotted into the shift.
  {
    Scalar<DataVector> shift_dot_normal{};
    get(shift_dot_normal)
        .set_data_ref(make_not_null(&get<1>(*packaged_char_speeds)));
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    get(shift_dot_normal) *= -1.0;
    // the metric mode speed is the zero speed times (1 + gamma_1)
    get<0>(*packaged_char_speeds) =
        (1.0 + get(constraint_gamma1)) * get(shift_dot_normal);
    // 2 = plus, 3 = minus
    get<2>(*packaged_char_speeds) = get(lapse) + get(shift_dot_normal);
    get<3>(*packaged_char_speeds) = -get(lapse) + get(shift_dot_normal);
  }

  if (normal_dot_mesh_velocity.has_value()) {
    get<0>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
    get<1>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
    get<2>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
    get<3>(*packaged_char_speeds) -= get(*normal_dot_mesh_velocity);
  }

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = a; b < Dim + 1; ++b) {
      packaged_char_speed_gamma2_v_spacetime_metric->get(a, b) =
          get(constraint_gamma2) * spacetime_metric.get(a, b);
    }
  }

  // Computes the contribution to the boundary correction from one side of the
  // interface.
  //
  // Note: when UpwindPenalty::dg_boundary_terms() is called, an Element passes
  // in its own packaged data to fill the interior fields, and its neighbor's
  // packaged data to fill the exterior fields. This introduces a sign flip for
  // each normal used in computing the exterior fields.
  // Use v_psi allocation as n^i Phi_i
  {
    tnsr::aa<DataVector, Dim, Frame::Inertial>& normal_dot_phi =
        *packaged_char_speed_v_spacetime_metric;
    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = a; b < Dim + 1; ++b) {
        normal_dot_phi.get(a, b) = get<0>(normal_vector) * phi.get(0, a, b);
        for (size_t i = 1; i < Dim; ++i) {
          normal_dot_phi.get(a, b) += normal_vector.get(i) * phi.get(i, a, b);
        }
      }
    }

    for (size_t a = 0; a < Dim + 1; ++a) {
      for (size_t b = a; b < Dim + 1; ++b) {
        packaged_char_speed_v_plus->get(a, b) =
            get<2>(*packaged_char_speeds) *
            (pi.get(a, b) + normal_dot_phi.get(a, b) -
             packaged_char_speed_gamma2_v_spacetime_metric->get(a, b));
        packaged_char_speed_v_minus->get(a, b) =
            get<3>(*packaged_char_speeds) *
            (pi.get(a, b) - normal_dot_phi.get(a, b) -
             packaged_char_speed_gamma2_v_spacetime_metric->get(a, b));

        for (size_t i = 0; i < Dim; ++i) {
          packaged_char_speed_v_zero->get(i, a, b) =
              get<1>(*packaged_char_speeds) *
              (phi.get(i, a, b) -
               normal_covector.get(i) * normal_dot_phi.get(a, b));
        }
      }
    }
  }

  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = a; b < Dim + 1; ++b) {
      for (size_t d = 0; d < Dim; ++d) {
        packaged_char_speed_n_times_v_plus->get(d, a, b) =
            packaged_char_speed_v_plus->get(a, b) * normal_covector.get(d);
        packaged_char_speed_n_times_v_minus->get(d, a, b) =
            packaged_char_speed_v_minus->get(a, b) * normal_covector.get(d);
      }
      packaged_char_speed_v_spacetime_metric->get(a, b) =
          get<0>(*packaged_char_speeds) * spacetime_metric.get(a, b);
      packaged_char_speed_gamma2_v_spacetime_metric->get(a, b) *=
          get<0>(*packaged_char_speeds);
    }
  }

  return max(max(get<0>(*packaged_char_speeds), get<1>(*packaged_char_speeds),
                 get<2>(*packaged_char_speeds), get<3>(*packaged_char_speeds)));
}

template <size_t Dim>
void UpwindPenalty<Dim>::dg_boundary_terms(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_phi,

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
        char_speed_normal_times_v_plus_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>&
        char_speed_normal_times_v_minus_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>&
        char_speed_constraint_gamma2_v_spacetime_metric_ext,
    const tnsr::a<DataVector, 3, Frame::Inertial>& char_speeds_ext,
    dg::Formulation /*dg_formulation*/) const noexcept {
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

  // D_spacetime_metric = Theta(-lambda_spacetime_metric^{ext})
  // lambda_spacetime_metric^{ext} v_spacetime_metric^{ext}
  //       - Theta(-lambda_spacetime_metric^{int}) lambda_spacetime_metric^{int}
  //       v_spacetime_metric^{int}
  // where the unit normals on both sides point in the same direction, out
  // of the current element. Since lambda_spacetime_metric from the neighbor is
  // computing with the normal vector pointing into the current element in the
  // code, we need to swap the sign of lambda_spacetime_metric^{ext}.
  for (size_t a = 0; a < Dim + 1; ++a) {
    for (size_t b = a; b < Dim + 1; ++b) {
      boundary_correction_spacetime_metric->get(a, b) =
          weighted_lambda_spacetime_metric_ext *
              char_speed_v_spacetime_metric_ext.get(a, b) -
          weighted_lambda_spacetime_metric_int *
              char_speed_v_spacetime_metric_int.get(a, b);

      boundary_correction_pi->get(a, b) =
          0.5 * (weighted_lambda_plus_ext * char_speed_v_plus_ext.get(a, b) +
                 weighted_lambda_minus_ext * char_speed_v_minus_ext.get(a, b)) +
          weighted_lambda_spacetime_metric_ext *
              char_speed_constraint_gamma2_v_spacetime_metric_ext.get(a, b)

          -
          0.5 * (weighted_lambda_plus_int * char_speed_v_plus_int.get(a, b) +
                 weighted_lambda_minus_int * char_speed_v_minus_int.get(a, b)) -
          weighted_lambda_spacetime_metric_int *
              char_speed_constraint_gamma2_v_spacetime_metric_int.get(a, b);

      for (size_t d = 0; d < Dim; ++d) {
        // Overall minus sign on ext because of normal vector is opposite
        // direction.
        boundary_correction_phi->get(d, a, b) =
            -0.5 * (weighted_lambda_minus_ext *
                        char_speed_normal_times_v_minus_ext.get(d, a, b) -
                    weighted_lambda_plus_ext *
                        char_speed_normal_times_v_plus_ext.get(d, a, b)) +
            weighted_lambda_zero_ext * char_speed_v_zero_ext.get(d, a, b)

            - 0.5 * (weighted_lambda_plus_int *
                         char_speed_normal_times_v_plus_int.get(d, a, b) -
                     weighted_lambda_minus_int *
                         char_speed_normal_times_v_minus_int.get(d, a, b)) -
            weighted_lambda_zero_int * char_speed_v_zero_int.get(d, a, b);
      }
    }
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
}  // namespace GeneralizedHarmonic::BoundaryCorrections
