// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Flattener.hpp"

#include <array>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace grmhd::ValenciaDivClean::Limiters {

FlattenerAction flatten_solution(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian) noexcept {
  // Create a temporary variable for each field's cell average.
  // These temporaries live on the stack and should have minimal cost.
  double mean_tilde_d = std::numeric_limits<double>::signaling_NaN();
  double mean_tilde_tau = std::numeric_limits<double>::signaling_NaN();
  auto mean_tilde_s =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());

  const auto compute_means = [&mean_tilde_d, &mean_tilde_tau, &mean_tilde_s,
                              &tilde_d, &tilde_tau, &tilde_s, &mesh,
                              &det_logical_to_inertial_jacobian]() noexcept {
    // Compute the means w.r.t. the inertial coords
    // (Note that several other parts of the limiter code take means w.r.t. the
    // logical coords, and therefore might not be conservative on curved grids)
    const double volume_of_cell =
        definite_integral(get(det_logical_to_inertial_jacobian), mesh);
    const auto inertial_coord_mean =
        [&mesh, &det_logical_to_inertial_jacobian,
         &volume_of_cell](const DataVector& u) noexcept {
          // Note that the term `det_jac * u` below results in an allocation.
          // If this function needs to be optimized, a buffer for the product
          // could be allocated outside the lambda, and updated in the lambda.
          return definite_integral(get(det_logical_to_inertial_jacobian) * u,
                                   mesh) /
                 volume_of_cell;
        };
    mean_tilde_d = inertial_coord_mean(get(*tilde_d));
    mean_tilde_tau = inertial_coord_mean(get(*tilde_tau));
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(mean_tilde_s, i) = inertial_coord_mean(tilde_s->get(i));
    }

    // sanity check the means
    ASSERT(mean_tilde_d > 0., "Invalid TildeD input to flattener");
  };

  FlattenerAction flattener_action = FlattenerAction::NoOp;

  // If min(tilde_d) is negative, then flatten.
  const double min_tilde_d = min(get(*tilde_d));
  if (min_tilde_d < 0.) {
    compute_means();

    // Note: the current algorithm flattens all fields by the same factor,
    // though in principle a different factor could be applied to each field.
    constexpr double safety = 0.95;
    const double factor = safety * mean_tilde_d / (mean_tilde_d - min_tilde_d);

    get(*tilde_d) = mean_tilde_d + factor * (get(*tilde_d) - mean_tilde_d);
    get(*tilde_tau) =
        mean_tilde_tau + factor * (get(*tilde_tau) - mean_tilde_tau);
    for (size_t i = 0; i < 3; ++i) {
      tilde_s->get(i) = gsl::at(mean_tilde_s, i) +
                        factor * (tilde_s->get(i) - gsl::at(mean_tilde_s, i));
    }

    flattener_action = FlattenerAction::ScaledSolution;
  }

  const Scalar<DataVector> tilde_b_squared =
      dot_product(tilde_b, tilde_b, spatial_metric);

  // Check TildeTau with the condition from Foucart's thesis
  bool need_to_flatten = false;
  for (size_t s = 0; s < mesh.number_of_grid_points(); ++s) {
    if (get(tilde_b_squared)[s] >
        2. * get(*tilde_tau)[s] * get(sqrt_det_spatial_metric)[s]) {
      need_to_flatten = true;
      break;
    }

    // TODO: consider adding the S^2 < S_{max}^2 check
    // but S_{max} is hard to compute, so is this worth it?
  }

  if (need_to_flatten) {
    if (flattener_action == FlattenerAction::NoOp) {
      // We didn't previously correct for negative TildeD, therefore the
      // means have not yet been computed
      compute_means();
    }

    get(*tilde_d) = mean_tilde_d;
    get(*tilde_tau) = mean_tilde_tau;
    for (size_t i = 0; i < 3; ++i) {
      tilde_s->get(i) = gsl::at(mean_tilde_s, i);
    }

    flattener_action = FlattenerAction::SetSolutionToMean;
  }

  return flattener_action;
}

}  // namespace grmhd::ValenciaDivClean::Limiters
