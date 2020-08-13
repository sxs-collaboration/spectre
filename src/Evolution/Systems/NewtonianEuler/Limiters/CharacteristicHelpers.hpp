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
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
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
    const tnsr::i<double, VolumeDim>& unit_normal) noexcept;

// @{
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
// @}

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

}  // namespace NewtonianEuler::Limiters
