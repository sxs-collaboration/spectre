// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Limiters/Flattener.hpp"

#include <array>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace NewtonianEuler::Limiters {

template <size_t VolumeDim, size_t ThermodynamicDim>
FlattenerAction flatten_solution(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh,
    const Scalar<DataVector>& det_logical_to_inertial_jacobian,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  // A note on the design behind the handling of the cell-averaged fields:
  //
  // The cell averages are needed for
  // - sanity-checking the inputs for validity
  // - flattening in case of negative density
  // - flattening in case of negative pressure
  //
  // In addition, we have the following optimization goals:
  // - cell averages are only computed once
  // - cell averages are only computed when needed
  //
  // To meet these goals, we create a temporary variable for each field's cell
  // average. These temporaries live on the stack and should have minimal cost.
  double mean_density = std::numeric_limits<double>::signaling_NaN();
  auto mean_momentum =
      make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());
  double mean_energy = std::numeric_limits<double>::signaling_NaN();

  const auto compute_means = [&mean_density, &mean_momentum, &mean_energy,
                              &mass_density_cons, &momentum_density,
                              &energy_density, &mesh,
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
    mean_density = inertial_coord_mean(get(*mass_density_cons));
    for (size_t i = 0; i < VolumeDim; ++i) {
      gsl::at(mean_momentum, i) = inertial_coord_mean(momentum_density->get(i));
    }
    mean_energy = inertial_coord_mean(get(*energy_density));

    // sanity check the means
    ASSERT(mean_density > 0., "Invalid mass density input to flattener");
    if constexpr (ThermodynamicDim == 2) {
      ASSERT(mean_energy > 0., "Invalid energy density input to flattener");
    }
  };

  FlattenerAction flattener_action = FlattenerAction::NoOp;

  // If min(density) is negative, then flatten.
  const double min_density = min(get(*mass_density_cons));
  if (min_density < 0.) {
    compute_means();

    // Note: the current algorithm flattens all fields by the same factor,
    // though in principle a different factor could be applied to each field.
    constexpr double safety = 0.95;
    const double factor = safety * mean_density / (mean_density - min_density);

    get(*mass_density_cons) =
        mean_density + factor * (get(*mass_density_cons) - mean_density);
    for (size_t i = 0; i < VolumeDim; ++i) {
      momentum_density->get(i) =
          gsl::at(mean_momentum, i) +
          factor * (momentum_density->get(i) - gsl::at(mean_momentum, i));
    }
    get(*energy_density) =
        mean_energy + factor * (get(*energy_density) - mean_energy);

    flattener_action = FlattenerAction::ScaledSolution;
  }

  // Check for negative pressures
  if constexpr (ThermodynamicDim == 2) {
    const auto specific_internal_energy = Scalar<DataVector>{
        get(*energy_density) / get(*mass_density_cons) -
        0.5 * get(dot_product(*momentum_density, *momentum_density)) /
            square(get(*mass_density_cons))};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        *mass_density_cons, specific_internal_energy);

    // If min(pressure) is negative, set solution to cell averages
    const double min_pressure = min(get(pressure));
    if (min_pressure < 0.) {
      if (flattener_action == FlattenerAction::NoOp) {
        // We didn't previously correct for negative densities, therefore the
        // means have not yet been computed
        compute_means();
      }

      get(*mass_density_cons) = mean_density;
      for (size_t i = 0; i < VolumeDim; ++i) {
        momentum_density->get(i) = gsl::at(mean_momentum, i);
      }
      get(*energy_density) = mean_energy;

      flattener_action = FlattenerAction::SetSolutionToMean;
    }
  }

  return flattener_action;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                      \
  template FlattenerAction flatten_solution(                      \
      gsl::not_null<Scalar<DataVector>*>,                         \
      gsl::not_null<tnsr::I<DataVector, DIM(data)>*>,             \
      gsl::not_null<Scalar<DataVector>*>, const Mesh<DIM(data)>&, \
      const Scalar<DataVector>&,                                  \
      const EquationsOfState::EquationOfState<false,              \
                                              THERMODIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef INSTANTIATE
#undef THERMODIM
#undef DIM

}  // namespace NewtonianEuler::Limiters
