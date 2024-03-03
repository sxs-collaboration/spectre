// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Flattener.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace grmhd::ValenciaDivClean {
template <typename RecoverySchemesList>
Flattener<RecoverySchemesList>::Flattener(
    const bool require_positive_mean_tilde_d,
    const bool require_positive_mean_tilde_ye,
    const bool require_physical_mean_tilde_tau, const bool recover_primitives)
    : require_positive_mean_tilde_d_(require_positive_mean_tilde_d),
      require_positive_mean_tilde_ye_(require_positive_mean_tilde_ye),
      require_physical_mean_tilde_tau_(require_physical_mean_tilde_tau),
      recover_primitives_(recover_primitives) {}

template <typename RecoverySchemesList>
void Flattener<RecoverySchemesList>::pup(PUP::er& p) {
  p | require_positive_mean_tilde_d_;
  p | require_positive_mean_tilde_ye_;
  p | require_physical_mean_tilde_tau_;
  p | recover_primitives_;
}

template <typename RecoverySchemesList>
void Flattener<RecoverySchemesList>::operator()(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> primitives,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_logical_to_inertial_inv_jacobian,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
        primitive_from_conservative_options) const {
  // Create a temporary variable for each field's cell average.
  // These temporaries live on the stack and should have minimal cost.
  double mean_tilde_d = std::numeric_limits<double>::signaling_NaN();
  double mean_tilde_ye = std::numeric_limits<double>::signaling_NaN();
  double mean_tilde_tau = std::numeric_limits<double>::signaling_NaN();
  auto mean_tilde_s =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  const Scalar<DataVector> det_logical_to_inertial_jacobian{
      DataVector{get(det_logical_to_inertial_inv_jacobian)}};
  bool already_computed_means = false;

  const auto compute_means = [
    &already_computed_means, &mean_tilde_d, &mean_tilde_ye, &mean_tilde_tau,
    &mean_tilde_s, &tilde_d, &tilde_ye, &tilde_tau, &tilde_s, &mesh,
    &det_logical_to_inertial_jacobian,
    require_positive_mean_tilde_d = require_positive_mean_tilde_d_,
    require_positive_mean_tilde_ye = require_positive_mean_tilde_ye_
  ]() {
    if (already_computed_means) {
      return;
    }
    already_computed_means = true;
    // Compute the means w.r.t. the inertial coords
    // (Note that several other parts of the limiter code take means w.r.t.
    // the logical coords, and therefore might not be conservative on curved
    // grids)
    const double volume_of_cell =
        definite_integral(get(det_logical_to_inertial_jacobian), mesh);
    const auto inertial_coord_mean = [&mesh, &det_logical_to_inertial_jacobian,
                                      &volume_of_cell](const DataVector& u) {
      // Note that the term `det_jac * u` below results in an
      // allocation. If this function needs to be optimized, a buffer
      // for the product could be allocated outside the lambda, and
      // updated in the lambda.
      return definite_integral(get(det_logical_to_inertial_jacobian) * u,
                               mesh) /
             volume_of_cell;
    };
    mean_tilde_d = inertial_coord_mean(get(*tilde_d));
    mean_tilde_ye = inertial_coord_mean(get(*tilde_ye));
    mean_tilde_tau = inertial_coord_mean(get(*tilde_tau));
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(mean_tilde_s, i) = inertial_coord_mean(tilde_s->get(i));
    }

    if (require_positive_mean_tilde_d and mean_tilde_d < 0.) {
      ERROR("We require TildeD to have positive mean, but got "
            << *tilde_d << " with mean value " << mean_tilde_d);
    }

    if (require_positive_mean_tilde_ye and mean_tilde_ye < 0.) {
      ERROR("We require TildeYe to have positive mean, but got "
            << *tilde_ye << " with mean value " << mean_tilde_ye);
    }
  };

  // If min(tilde_d) is negative, then flatten.
  if (const double min_tilde_d = min(get(*tilde_d)),
      min_tilde_ye = min(get(*tilde_ye));
      min_tilde_d < 0. or min_tilde_ye < 0.) {
    compute_means();

    // Note: the current algorithm flattens all fields by the same factor,
    // though in principle a different factor could be applied to each field.
    //
    // Experiments with cylindrical blast wave simulations showed that applying
    // the same factor to all fields here dramatically improves results for the
    // cylindrical blast wave. Of course, this might differ depending on the
    // test problem.
    constexpr double safety = 0.95;
    const double factor = safety * mean_tilde_d / (mean_tilde_d - min_tilde_d);

    get(*tilde_d) = mean_tilde_d + factor * (get(*tilde_d) - mean_tilde_d);
    get(*tilde_ye) = mean_tilde_ye + factor * (get(*tilde_ye) - mean_tilde_ye);
    get(*tilde_tau) =
        mean_tilde_tau + factor * (get(*tilde_tau) - mean_tilde_tau);
    for (size_t i = 0; i < 3; ++i) {
      tilde_s->get(i) = gsl::at(mean_tilde_s, i) +
                        factor * (tilde_s->get(i) - gsl::at(mean_tilde_s, i));
    }
  }

  const Scalar<DataVector> tilde_b_squared =
      dot_product(tilde_b, tilde_b, spatial_metric);

  if (require_physical_mean_tilde_tau_) {
    compute_means();
    if (UNLIKELY(max((0.5 / mean_tilde_tau) * get(tilde_b_squared) /
                     get(sqrt_det_spatial_metric)) > 1.0)) {
      ERROR(
          "Unable to rescale TildeTau to the required range in a "
          "conservative manner:\n"
          "mean_tilde_tau: "
          << mean_tilde_tau << "\ntilde_b_squared: " << tilde_b_squared
          << "\nsqrt_det_spatial_metric: " << sqrt_det_spatial_metric);
    }
  }

  // Check TildeTau with the condition from Foucart's thesis
  //
  // Increase tilde_tau if necessary. We do this by _decreasing_ the amount of
  // oscillation around the mean.
  compute_means();
  const size_t num_pts = get(*tilde_tau).size();
  double tilde_tau_scale_factor = 1.;
  for (size_t i = 0; i < num_pts; ++i) {
    if (0.5 * get(tilde_b_squared)[i] / get(sqrt_det_spatial_metric)[i] /
            get(*tilde_tau)[i] >
        1.) {
      tilde_tau_scale_factor =
          std::min(tilde_tau_scale_factor,
                   std::abs((0.5 * get(tilde_b_squared)[i] /
                                 get(sqrt_det_spatial_metric)[i] -
                             mean_tilde_tau) /
                            (get(*tilde_tau)[i] - mean_tilde_tau)));
    }
  }
  if (tilde_tau_scale_factor < 1.) {
    constexpr double safety = 0.99;
    get(*tilde_tau) = mean_tilde_tau + (safety * tilde_tau_scale_factor) *
                                           (get(*tilde_tau) - mean_tilde_tau);
  }

  // Check if we can recover the primitive variables. If not, then we need to
  // flatten.
  const size_t number_of_points = mesh.number_of_grid_points();
  Variables<hydro::grmhd_tags<DataVector>> temp_prims(number_of_points);
  get<hydro::Tags::Pressure<DataVector>>(temp_prims) =
      get<hydro::Tags::Pressure<DataVector>>(*primitives);
  if (not grmhd::ValenciaDivClean::
          PrimitiveFromConservative<RecoverySchemesList, false>::apply(
              make_not_null(
                  &get<hydro::Tags::RestMassDensity<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::ElectronFraction<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                      temp_prims)),
              make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                  temp_prims)),
              make_not_null(
                  &get<hydro::Tags::MagneticField<DataVector, 3>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::DivergenceCleaningField<DataVector>>(
                      temp_prims)),
              make_not_null(
                  &get<hydro::Tags::LorentzFactor<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::Pressure<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::Temperature<DataVector>>(temp_prims)),
              *tilde_d, *tilde_ye, *tilde_tau, *tilde_s, tilde_b, tilde_phi,
              spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric, eos,
              primitive_from_conservative_options)) {
    compute_means();

    get(*tilde_d) = mean_tilde_d;
    get(*tilde_ye) = mean_tilde_ye;
    get(*tilde_tau) = mean_tilde_tau;
    for (size_t i = 0; i < 3; ++i) {
      tilde_s->get(i) = gsl::at(mean_tilde_s, i);
    }

    if (recover_primitives_) {
      grmhd::ValenciaDivClean::
          PrimitiveFromConservative<RecoverySchemesList, true>::apply(
              make_not_null(
                  &get<hydro::Tags::RestMassDensity<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::ElectronFraction<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                      temp_prims)),
              make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
                  temp_prims)),
              make_not_null(
                  &get<hydro::Tags::MagneticField<DataVector, 3>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::DivergenceCleaningField<DataVector>>(
                      temp_prims)),
              make_not_null(
                  &get<hydro::Tags::LorentzFactor<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::Pressure<DataVector>>(temp_prims)),
              make_not_null(
                  &get<hydro::Tags::Temperature<DataVector>>(temp_prims)),
              *tilde_d, *tilde_ye, *tilde_tau, *tilde_s, tilde_b, tilde_phi,
              spatial_metric, inv_spatial_metric, sqrt_det_spatial_metric, eos,
              primitive_from_conservative_options);
    }
  }
  if (recover_primitives_) {
    using std::swap;
    swap(*primitives, temp_prims);
  }
}

template <typename RecoverySchemesList>
bool operator==(const Flattener<RecoverySchemesList>& lhs,
                const Flattener<RecoverySchemesList>& rhs) {
  return lhs.require_positive_mean_tilde_d_ ==
             rhs.require_positive_mean_tilde_d_ and
         lhs.require_positive_mean_tilde_ye_ ==
             rhs.require_positive_mean_tilde_ye_ and
         lhs.require_physical_mean_tilde_tau_ ==
             rhs.require_physical_mean_tilde_tau_ and
         lhs.recover_primitives_ == rhs.recover_primitives_;
}

template <typename RecoverySchemesList>
bool operator!=(const Flattener<RecoverySchemesList>& lhs,
                const Flattener<RecoverySchemesList>& rhs) {
  return not(lhs == rhs);
}

using NewmanHamlinThenPalenzuelaEtAl =
    tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl,
               PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;

#define ORDERED_RECOVERY_LIST                            \
  (tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl>,    \
   tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin>,   \
   tmpl::list<PrimitiveRecoverySchemes::PalenzuelaEtAl>, \
   NewmanHamlinThenPalenzuelaEtAl, KastaunThenNewmanThenPalenzuela)

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                    \
  template class Flattener<RECOVERY(data)>;                       \
  template bool operator==(const Flattener<RECOVERY(data)>& lhs,  \
                           const Flattener<RECOVERY(data)>& rhs); \
  template bool operator!=(const Flattener<RECOVERY(data)>& lhs,  \
                           const Flattener<RECOVERY(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, ORDERED_RECOVERY_LIST)
#undef INSTANTIATION

#undef THERMO_DIM
#undef RECOVERY
#undef ORDERED_RECOVERY_LIST
}  // namespace grmhd::ValenciaDivClean
