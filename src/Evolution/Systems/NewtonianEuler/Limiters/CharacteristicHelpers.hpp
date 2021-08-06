// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace NewtonianEuler::Limiters {

/*!
 * \brief Compute the transform matrices between the conserved variables and
 * the characteristic variables of the NewtonianEuler system.
 *
 * Wraps calls to `NewtonianEuler::right_eigenvectors` and
 * `NewtonianEuler::left_eigenvectors`.
 */
template <size_t VolumeDim, size_t ThermodynamicDim>
std::pair<Matrix, Matrix> right_and_left_eigenvectors(
    const Scalar<double>& mean_density,
    const tnsr::I<double, VolumeDim>& mean_momentum,
    const Scalar<double>& mean_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const tnsr::i<double, VolumeDim>& unit_vector,
    bool compute_char_transformation_numerically = false) noexcept;

/// @{
/// \brief Compute characteristic fields from conserved fields
///
/// Note that these functions apply the same transformation to every grid point
/// in the element, using the same matrix from `left_eigenvectors`.
/// This is in contrast to the characteristic transformation used in the
/// GeneralizedHarmonic upwind flux, which is computed pointwise.
///
/// By using a fixed matrix of eigenvectors computed from the cell-averaged
/// fields, we ensure we can consistently transform back from the characteristic
/// variables even after applying the limiter (the limiter changes the pointwise
/// values but preserves the cell averages).
template <size_t VolumeDim>
void characteristic_fields(
    gsl::not_null<tuples::TaggedTuple<
        ::Tags::Mean<NewtonianEuler::Tags::VMinus>,
        ::Tags::Mean<NewtonianEuler::Tags::VMomentum<VolumeDim>>,
        ::Tags::Mean<NewtonianEuler::Tags::VPlus>>*>
        char_means,
    const tuples::TaggedTuple<
        ::Tags::Mean<NewtonianEuler::Tags::MassDensityCons>,
        ::Tags::Mean<NewtonianEuler::Tags::MomentumDensity<VolumeDim>>,
        ::Tags::Mean<NewtonianEuler::Tags::EnergyDensity>>& cons_means,
    const Matrix& left) noexcept;

template <size_t VolumeDim>
void characteristic_fields(
    gsl::not_null<Scalar<DataVector>*> char_v_minus,
    gsl::not_null<tnsr::I<DataVector, VolumeDim>*> char_v_momentum,
    gsl::not_null<Scalar<DataVector>*> char_v_plus,
    const Scalar<DataVector>& cons_mass_density,
    const tnsr::I<DataVector, VolumeDim>& cons_momentum_density,
    const Scalar<DataVector>& cons_energy_density, const Matrix& left) noexcept;

template <size_t VolumeDim>
void characteristic_fields(
    gsl::not_null<
        Variables<tmpl::list<NewtonianEuler::Tags::VMinus,
                             NewtonianEuler::Tags::VMomentum<VolumeDim>,
                             NewtonianEuler::Tags::VPlus>>*>
        char_vars,
    const Variables<tmpl::list<NewtonianEuler::Tags::MassDensityCons,
                               NewtonianEuler::Tags::MomentumDensity<VolumeDim>,
                               NewtonianEuler::Tags::EnergyDensity>>& cons_vars,
    const Matrix& left) noexcept;
/// @}

/// \brief Compute conserved fields from characteristic fields
///
/// Note that this function applies the same transformation to every grid point
/// in the element, using the same matrix from `right_eigenvectors`.
/// This is in contrast to the characteristic transformation used in the
/// GeneralizedHarmonic upwind flux, which is computed pointwise.
///
/// By using a fixed matrix of eigenvectors computed from the cell-averaged
/// fields, we ensure we can consistently transform back from the characteristic
/// variables even after applying the limiter (the limiter changes the pointwise
/// values but preserves the cell averages).
template <size_t VolumeDim>
void conserved_fields_from_characteristic_fields(
    gsl::not_null<Scalar<DataVector>*> cons_mass_density,
    gsl::not_null<tnsr::I<DataVector, VolumeDim>*> cons_momentum_density,
    gsl::not_null<Scalar<DataVector>*> cons_energy_density,
    const Scalar<DataVector>& char_v_minus,
    const tnsr::I<DataVector, VolumeDim>& char_v_momentum,
    const Scalar<DataVector>& char_v_plus, const Matrix& right) noexcept;

/// \brief Apply a limiter to the characteristic fields computed with respect
/// to each direction in the volume, then take average of results
///
/// When computing the characteristic fields in the volume to pass as inputs to
/// the limiter, it is not necessarily clear (in more than one dimension) which
/// unit vector should be used in the characteristic decomposition. This is in
/// contrast to uses of characteristic fields for, e.g., computing boundary
/// conditions where the boundary normal provides a clear choice of unit vector.
//
/// The common solution to this challenge is to average the results of applying
/// the limiter with different choices of characteristic decomposition:
/// - For each direction (e.g. in 3D for each of
///   \f$(\hat{x}, \hat{y}, \hat{z})\f$), compute the characteristic fields with
///   respect to this direction. This gives `VolumeDim` sets of characteristic
///   fields.
/// - Apply the limiter to each set of characteristic fields, then convert back
///   to conserved fields. This gives `VolumeDim` different sets of limited
///   conserved fields.
/// - Average the new conserved fields; this is the result of the limiting
///   process.
///
/// This function handles the logic of computing the different characteristic
/// fields, limiting them, converting back to conserved fields, and averaging.
///
/// The limiter to apply is passed in via a lambda which is responsible for
/// converting any neighbor data to the characteristic representation, and then
/// applying the limiter to all NewtonianEuler characteristic fields.
template <size_t VolumeDim, size_t ThermodynamicDim, typename LimiterLambda>
bool apply_limiter_to_characteristic_fields_in_all_directions(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, VolumeDim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Mesh<VolumeDim>& mesh,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const LimiterLambda& prepare_and_apply_limiter,
    const bool compute_char_transformation_numerically = false) noexcept {
  // Temp variables for calculations
  // There are quite a few tensors in this allocation because in general we need
  // to preserve the input (cons field) tensors until they are overwritten at
  // the very end of the computation... then we need additional buffers for the
  // char fields, the limited cons fields, and the accumulated cons fields for
  // averaging.
  //
  // Possible optimization: specialize the 1D case which doesn't do average so
  // doesn't need as many buffers
  Variables<tmpl::list<
      NewtonianEuler::Tags::VMinus, NewtonianEuler::Tags::VMomentum<VolumeDim>,
      NewtonianEuler::Tags::VPlus, ::Tags::TempScalar<0>,
      ::Tags::TempI<0, VolumeDim>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempI<1, VolumeDim>, ::Tags::TempScalar<3>>>
      temp_buffer(mesh.number_of_grid_points());
  auto& char_v_minus = get<NewtonianEuler::Tags::VMinus>(temp_buffer);
  auto& char_v_momentum =
      get<NewtonianEuler::Tags::VMomentum<VolumeDim>>(temp_buffer);
  auto& char_v_plus = get<NewtonianEuler::Tags::VPlus>(temp_buffer);
  auto& temp_mass_density_cons = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& temp_momentum_density = get<::Tags::TempI<0, VolumeDim>>(temp_buffer);
  auto& temp_energy_density = get<::Tags::TempScalar<1>>(temp_buffer);
  auto& accumulate_mass_density_cons = get<::Tags::TempScalar<2>>(temp_buffer);
  auto& accumulate_momentum_density =
      get<::Tags::TempI<1, VolumeDim>>(temp_buffer);
  auto& accumulate_energy_density = get<::Tags::TempScalar<3>>(temp_buffer);

  // Initialize the accumulating tensors
  get(accumulate_mass_density_cons) = 0.;
  for (size_t i = 0; i < VolumeDim; ++i) {
    accumulate_momentum_density.get(i) = 0.;
  }
  get(accumulate_energy_density) = 0.;

  // Cellwise means, used in computing the cons/char transformations
  const auto mean_density =
      Scalar<double>{mean_value(get(*mass_density_cons), mesh)};
  const auto mean_momentum = [&momentum_density, &mesh]() noexcept {
    tnsr::I<double, VolumeDim> result{};
    for (size_t i = 0; i < VolumeDim; ++i) {
      result.get(i) = mean_value(momentum_density->get(i), mesh);
    }
    return result;
  }();
  const auto mean_energy =
      Scalar<double>{mean_value(get(*energy_density), mesh)};

  bool some_component_was_limited = false;

  // Loop over directions, then compute chars w.r.t. this direction and limit
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto unit_vector = [&d]() noexcept {
      auto components = make_array<VolumeDim>(0.);
      components[d] = 1.;
      return tnsr::i<double, VolumeDim>(components);
    }();
    const auto right_and_left =
        NewtonianEuler::Limiters::right_and_left_eigenvectors(
            mean_density, mean_momentum, mean_energy, equation_of_state,
            unit_vector, compute_char_transformation_numerically);
    const auto& right = right_and_left.first;
    const auto& left = right_and_left.second;

    // Transform tensors to characteristics
    NewtonianEuler::Limiters::characteristic_fields(
        make_not_null(&char_v_minus), make_not_null(&char_v_momentum),
        make_not_null(&char_v_plus), *mass_density_cons, *momentum_density,
        *energy_density, left);

    // Transform neighbor data and apply limiter
    const bool some_component_was_limited_with_this_unit_vector =
        prepare_and_apply_limiter(make_not_null(&char_v_minus),
                                  make_not_null(&char_v_momentum),
                                  make_not_null(&char_v_plus), left);

    some_component_was_limited =
        some_component_was_limited_with_this_unit_vector or
        some_component_was_limited;

    // Transform back to conserved variables. But skip the transformation if no
    // limiting occured with this unit vector.
    if (some_component_was_limited_with_this_unit_vector) {
      NewtonianEuler::Limiters::conserved_fields_from_characteristic_fields(
          make_not_null(&temp_mass_density_cons),
          make_not_null(&temp_momentum_density),
          make_not_null(&temp_energy_density), char_v_minus, char_v_momentum,
          char_v_plus, right);
    } else {
      temp_mass_density_cons = *mass_density_cons;
      temp_momentum_density = *momentum_density;
      temp_energy_density = *energy_density;
    }

    // Add to running sum for averaging
    const double one_over_dim = 1.0 / static_cast<double>(VolumeDim);
    get(accumulate_mass_density_cons) +=
        one_over_dim * get(temp_mass_density_cons);
    for (size_t i = 0; i < VolumeDim; ++i) {
      accumulate_momentum_density.get(i) +=
          one_over_dim * temp_momentum_density.get(i);
    }
    get(accumulate_energy_density) += one_over_dim * get(temp_energy_density);
  }  // for loop over dimensions

  *mass_density_cons = accumulate_mass_density_cons;
  *momentum_density = accumulate_momentum_density;
  *energy_density = accumulate_energy_density;
  return some_component_was_limited;
}

}  // namespace NewtonianEuler::Limiters
