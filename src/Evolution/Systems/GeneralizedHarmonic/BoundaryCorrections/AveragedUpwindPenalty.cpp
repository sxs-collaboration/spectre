// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/AveragedUpwindPenalty.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gh::BoundaryCorrections {
template <size_t Dim>
AveragedUpwindPenalty<Dim>::AveragedUpwindPenalty(CkMigrateMessage* msg)
    : BoundaryCorrection(msg) {}

template <size_t Dim>
std::unique_ptr<evolution::BoundaryCorrection>
AveragedUpwindPenalty<Dim>::get_clone() const {
  return std::make_unique<AveragedUpwindPenalty>(*this);
}

template <size_t Dim>
void AveragedUpwindPenalty<Dim>::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
}

template <size_t Dim>
double AveragedUpwindPenalty<Dim>::dg_package_data(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        packaged_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        packaged_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_constraint_gamma1,
    const gsl::not_null<Scalar<DataVector>*> packaged_constraint_gamma2,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        packaged_normal,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        packaged_mesh_velocity,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,

    const Scalar<DataVector>& constraint_gamma1,
    const Scalar<DataVector>& constraint_gamma2,
    const Scalar<DataVector>& /*lapse*/,
    const tnsr::I<DataVector, Dim>& /*shift*/,

    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/)
    const {
  *packaged_spacetime_metric = spacetime_metric;
  *packaged_pi = pi;
  *packaged_phi = phi;
  *packaged_constraint_gamma1 = constraint_gamma1;
  *packaged_constraint_gamma2 = constraint_gamma2;
  *packaged_normal = normal_covector;
  if (mesh_velocity.has_value()) {
    *packaged_mesh_velocity = *mesh_velocity;
  } else {
    for (auto& component : *packaged_mesh_velocity) {
      component = 0.0;
    }
  }

  // This is supposed to return the max characteristic speed, but the
  // result is unused, so we don't do all the additional work required
  // just to throw it away.  We don't return NaN because the product
  // corrections for the combined systems return the max of the
  // results of each of the individual systems' corrections.  The max
  // double should be sufficiently obviously wrong that if anyone
  // actually tries to use it it will fail completely.
  return std::numeric_limits<double>::max();
}

template <size_t Dim>
void AveragedUpwindPenalty<Dim>::dg_boundary_terms(
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_pi,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, Frame::Inertial>*>
        boundary_correction_phi,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_int,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_int,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_int,
    const Scalar<DataVector>& constraint_gamma1_int,
    const Scalar<DataVector>& constraint_gamma2_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_int,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& mesh_velocity_int,

    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric_ext,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi_ext,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi_ext,
    const Scalar<DataVector>& constraint_gamma1_ext,
    const Scalar<DataVector>& constraint_gamma2_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_ext,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& mesh_velocity_ext,

    const dg::Formulation dg_formulation) const {
  if (dg_formulation != dg::Formulation::StrongInertial) {
    ERROR_NO_TRACE("AveragedUpwindPenalty only coded for StrongInertial form");
  }

  const size_t num_pts = get<0, 0>(spacetime_metric_int).size();
  Variables<tmpl::list<
      ::Tags::Tempaa<0, Dim>, ::Tags::Tempii<1, Dim>, ::Tags::TempII<2, Dim>,
      ::Tags::TempI<3, Dim>, ::Tags::TempScalar<4>, ::Tags::Tempi<5, Dim>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::Tempaa<11, Dim>>>
      buffer(num_pts);

  // Naming: Interior stuff is *_int, exterior stuff is *_ext, *_jump
  // is _ext - _int.  Other things are averages of the function
  // arguments or things computed from such averages.
  auto& spacetime_metric = get<::Tags::Tempaa<0, Dim>>(buffer);
  tenex::evaluate<ti::a, ti::b>(make_not_null(&spacetime_metric),
                                0.5 * (spacetime_metric_int(ti::a, ti::b) +
                                       spacetime_metric_ext(ti::a, ti::b)));
  auto& spatial_metric = get<::Tags::Tempii<1, Dim>>(buffer);
  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
  auto& inverse_spatial_metric = get<::Tags::TempII<2, Dim>>(buffer);
  {
    auto& determinant = get<::Tags::TempScalar<4>>(buffer);
    determinant_and_inverse(make_not_null(&determinant),
                            make_not_null(&inverse_spatial_metric),
                            spatial_metric);
  }
  auto& shift = get<::Tags::TempI<3, Dim>>(buffer);
  gr::shift(make_not_null(&shift), spacetime_metric, inverse_spatial_metric);
  auto& lapse = get<::Tags::TempScalar<4>>(buffer);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);
  auto& normal_covector = get<::Tags::Tempi<5, Dim>>(buffer);
  // Sign flip to make neighbor value outgoing for us.
  tenex::evaluate<ti::i>(make_not_null(&normal_covector),
                         0.5 * (normal_int(ti::i) - normal_ext(ti::i)));
  {
    auto& inverse_magnitude = get<::Tags::TempScalar<6>>(buffer);
    tenex::evaluate(make_not_null(&inverse_magnitude),
                    1.0 / sqrt(normal_covector(ti::i) *
                               inverse_spatial_metric(ti::I, ti::J) *
                               normal_covector(ti::j)));
    tenex::update<ti::i>(make_not_null(&normal_covector),
                         inverse_magnitude() * normal_covector(ti::i));
  }
  auto& constraint_gamma2 = get<::Tags::TempScalar<6>>(buffer);
  get(constraint_gamma2) =
      0.5 * (get(constraint_gamma2_int) + get(constraint_gamma2_ext));

  auto& incoming_char_speed_0 = get<::Tags::TempScalar<7>>(buffer);
  tenex::evaluate(make_not_null(&incoming_char_speed_0),
                  -normal_covector(ti::i) *
                      (shift(ti::I) + 0.5 * (mesh_velocity_int(ti::I) +
                                             mesh_velocity_ext(ti::I))));
  auto& incoming_char_speed_plus = get<::Tags::TempScalar<8>>(buffer);
  get(incoming_char_speed_plus) = get(incoming_char_speed_0) + get(lapse);
  auto& incoming_char_speed_minus = get<::Tags::TempScalar<9>>(buffer);
  get(incoming_char_speed_minus) = get(incoming_char_speed_0) - get(lapse);

  // Zero out the outgoing (average) speeds
  get(incoming_char_speed_0) *= step_function(-get(incoming_char_speed_0));
  get(incoming_char_speed_plus) *=
      step_function(-get(incoming_char_speed_plus));
  get(incoming_char_speed_minus) *=
      step_function(-get(incoming_char_speed_minus));

  auto& incoming_char_speed_g = get<::Tags::TempScalar<10>>(buffer);
  get(incoming_char_speed_g) =
      (1.0 + 0.5 * (get(constraint_gamma1_int) + get(constraint_gamma1_ext))) *
      get(incoming_char_speed_0);

  // lapse no longer needed, but scoping it to here would be hard, so
  // just have to be careful.
  auto& incoming_char_speed_sym = lapse;
  get(incoming_char_speed_sym) =
      0.5 * (get(incoming_char_speed_plus) + get(incoming_char_speed_minus));
  // _plus no longer needed except for computing _antisym, but again,
  // just have to be careful.
  auto& incoming_char_speed_antisym = incoming_char_speed_plus;
  get(incoming_char_speed_antisym) -= get(incoming_char_speed_minus);
  get(incoming_char_speed_antisym) *= 0.5;

  // The correction to the spatial_metric is proportional to the jump,
  // so store that here for convenience.
  auto& spacetime_metric_jump = *boundary_correction_spacetime_metric;
  tenex::evaluate<ti::a, ti::b>(
      make_not_null(&spacetime_metric_jump),
      spacetime_metric_ext(ti::a, ti::b) - spacetime_metric_int(ti::a, ti::b));

  auto& pi_jump = get<::Tags::Tempaa<11, Dim>>(buffer);
  tenex::evaluate<ti::a, ti::b>(make_not_null(&pi_jump),
                                pi_ext(ti::a, ti::b) - pi_int(ti::a, ti::b));

  // boundary_correction_phi holds phi_jump, briefly
  tenex::evaluate<ti::i, ti::a, ti::b>(
      boundary_correction_phi,
      phi_ext(ti::i, ti::a, ti::b) - phi_int(ti::i, ti::a, ti::b));
  // spacetime_metric no longer needed, but again, just have to be
  // careful.
  auto& normal_phi_jump = spacetime_metric;
  tenex::evaluate<ti::a, ti::b>(
      make_not_null(&normal_phi_jump),
      normal_covector(ti::i) * inverse_spatial_metric(ti::I, ti::J) *
          (*boundary_correction_phi)(ti::j, ti::a, ti::b));

  tenex::update<ti::i, ti::a, ti::b>(
      boundary_correction_phi,
      incoming_char_speed_0() *
              (*boundary_correction_phi)(ti::i, ti::a, ti::b) +
          normal_covector(ti::i) *
              ((incoming_char_speed_sym() - incoming_char_speed_0()) *
                   normal_phi_jump(ti::a, ti::b) +
               incoming_char_speed_antisym() *
                   (pi_jump(ti::a, ti::b) -
                    constraint_gamma2() *
                        spacetime_metric_jump(ti::a, ti::b))));

  tenex::evaluate<ti::a, ti::b>(
      boundary_correction_pi,
      (incoming_char_speed_g() - incoming_char_speed_sym()) *
              constraint_gamma2() * spacetime_metric_jump(ti::a, ti::b) +
          incoming_char_speed_sym() * pi_jump(ti::a, ti::b) +
          incoming_char_speed_antisym() * normal_phi_jump(ti::a, ti::b));

  // Recall that boundary_correction_spacetime_metric already holds the jump
  tenex::update<ti::a, ti::b>(
      boundary_correction_spacetime_metric,
      incoming_char_speed_g() *
          (*boundary_correction_spacetime_metric)(ti::a, ti::b));
}

template <size_t Dim>
bool operator==(const AveragedUpwindPenalty<Dim>& /*lhs*/,
                const AveragedUpwindPenalty<Dim>& /*rhs*/) {
  return true;
}

template <size_t Dim>
bool operator!=(const AveragedUpwindPenalty<Dim>& lhs,
                const AveragedUpwindPenalty<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID AveragedUpwindPenalty<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template class AveragedUpwindPenalty<DIM(data)>;                           \
  template bool operator==(const AveragedUpwindPenalty<DIM(data)>& /*lhs*/,  \
                           const AveragedUpwindPenalty<DIM(data)>& /*rhs*/); \
  template bool operator!=(const AveragedUpwindPenalty<DIM(data)>& /*lhs*/,  \
                           const AveragedUpwindPenalty<DIM(data)>& /*rhs*/);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace gh::BoundaryCorrections
