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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::ValenciaDivClean::Limiters {

/*!
 * \brief Compute the transform matrices between the conserved variables and
 * the characteristic variables of the ValenciaDivClean system.
 *
 * Wraps calls to `grmhd::ValenciaDivClean::numerical_eigensystem`.
 */
template <size_t ThermodynamicDim>
std::pair<Matrix, Matrix> right_and_left_eigenvectors(
    const Scalar<double>& mean_tilde_d, const Scalar<double>& mean_tilde_tau,
    const tnsr::i<double, 3>& mean_tilde_s,
    const tnsr::I<double, 3>& mean_tilde_b,
    const Scalar<double>& mean_tilde_phi, const Scalar<double>& mean_lapse,
    const tnsr::I<double, 3>& mean_shift,
    const tnsr::ii<double, 3>& mean_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const tnsr::i<double, 3>& unit_vector) noexcept;

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
void characteristic_fields(
    gsl::not_null<tuples::TaggedTuple<
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::VMinus>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::VMomentum>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::VPlus>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>*>
        char_means,
    const tuples::TaggedTuple<
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeD>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeTau>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeS<>>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeB<>>,
        ::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildePhi>>& cons_means,
    const Matrix& left) noexcept;

void characteristic_fields(
    gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
    gsl::not_null<Scalar<DataVector>*> char_v_minus,
    gsl::not_null<tnsr::I<DataVector, 5>*> char_v_momentum,
    gsl::not_null<Scalar<DataVector>*> char_v_plus,
    gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
    const Scalar<DataVector>& cons_tilde_d,
    const Scalar<DataVector>& cons_tilde_tau,
    const tnsr::i<DataVector, 3>& cons_tilde_s,
    const tnsr::I<DataVector, 3>& cons_tilde_b,
    const Scalar<DataVector>& cons_tilde_phi, const Matrix& left) noexcept;

void characteristic_fields(
    gsl::not_null<
        Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
                             grmhd::ValenciaDivClean::Tags::VMinus,
                             grmhd::ValenciaDivClean::Tags::VMomentum,
                             grmhd::ValenciaDivClean::Tags::VPlus,
                             grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>*>
        char_vars,
    const Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                               grmhd::ValenciaDivClean::Tags::TildeTau,
                               grmhd::ValenciaDivClean::Tags::TildeS<>,
                               grmhd::ValenciaDivClean::Tags::TildeB<>,
                               grmhd::ValenciaDivClean::Tags::TildePhi>>&
        cons_vars,
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
void conserved_fields_from_characteristic_fields(
    gsl::not_null<Scalar<DataVector>*> cons_tilde_d,
    gsl::not_null<Scalar<DataVector>*> cons_tilde_tau,
    gsl::not_null<tnsr::i<DataVector, 3>*> cons_tilde_s,
    gsl::not_null<tnsr::I<DataVector, 3>*> cons_tilde_b,
    gsl::not_null<Scalar<DataVector>*> cons_tilde_phi,
    const Scalar<DataVector>& char_v_div_clean_minus,
    const Scalar<DataVector>& char_v_minus,
    const tnsr::I<DataVector, 5>& char_v_momentum,
    const Scalar<DataVector>& char_v_plus,
    const Scalar<DataVector>& char_v_div_clean_plus,
    const Matrix& right) noexcept;

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
///   respect to this direction. This gives 3 sets of characteristic fields.
/// - Apply the limiter to each set of characteristic fields, then convert back
///   to conserved fields. This gives 3 different sets of limited conserved
///   fields.
/// - Average the new conserved fields; this is the result of the limiting
///   process.
///
/// This function handles the logic of computing the different characteristic
/// fields, limiting them, converting back to conserved fields, and averaging.
///
/// The limiter to apply is passed in via a lambda which is responsible for
/// converting any neighbor data to the characteristic representation, and then
/// applying the limiter to all ValenciaDivClean characteristic fields.
template <size_t ThermodynamicDim, typename LimiterLambda>
bool apply_limiter_to_characteristic_fields_in_all_directions(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3>& spatial_metric, const Mesh<3>& mesh,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const LimiterLambda& prepare_and_apply_limiter) noexcept {
  // Temp variables for calculations
  // There are quite a few tensors in this allocation because in general we need
  // to preserve the input (cons field) tensors until they are overwritten at
  // the very end of the computation... then we need additional buffers for the
  // char fields, the limited cons fields, and the accumulated cons fields for
  // averaging.
  Variables<tmpl::list<
      grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
      grmhd::ValenciaDivClean::Tags::VMinus,
      grmhd::ValenciaDivClean::Tags::VMomentum,
      grmhd::ValenciaDivClean::Tags::VPlus,
      grmhd::ValenciaDivClean::Tags::VDivCleanPlus,

      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::Tempi<0, 3>,
      ::Tags::TempI<0, 3>, ::Tags::TempScalar<2>,

      ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::Tempi<1, 3>,
      ::Tags::TempI<1, 3>, ::Tags::TempScalar<5>>>
      temp_buffer(mesh.number_of_grid_points());
  auto& char_v_div_clean_minus =
      get<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>(temp_buffer);
  auto& char_v_minus = get<grmhd::ValenciaDivClean::Tags::VMinus>(temp_buffer);
  auto& char_v_momentum =
      get<grmhd::ValenciaDivClean::Tags::VMomentum>(temp_buffer);
  auto& char_v_plus = get<grmhd::ValenciaDivClean::Tags::VPlus>(temp_buffer);
  auto& char_v_div_clean_plus =
      get<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>(temp_buffer);

  auto& temp_tilde_d = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& temp_tilde_tau = get<::Tags::TempScalar<1>>(temp_buffer);
  auto& temp_tilde_s = get<::Tags::Tempi<0, 3>>(temp_buffer);
  auto& temp_tilde_b = get<::Tags::TempI<0, 3>>(temp_buffer);
  auto& temp_tilde_phi = get<::Tags::TempScalar<2>>(temp_buffer);

  auto& accumulate_tilde_d = get<::Tags::TempScalar<3>>(temp_buffer);
  auto& accumulate_tilde_tau = get<::Tags::TempScalar<4>>(temp_buffer);
  auto& accumulate_tilde_s = get<::Tags::Tempi<1, 3>>(temp_buffer);
  auto& accumulate_tilde_b = get<::Tags::TempI<1, 3>>(temp_buffer);
  auto& accumulate_tilde_phi = get<::Tags::TempScalar<5>>(temp_buffer);

  // Initialize the accumulating tensors
  get(accumulate_tilde_d) = 0.;
  get(accumulate_tilde_tau) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    accumulate_tilde_s.get(i) = 0.;
    accumulate_tilde_b.get(i) = 0.;
  }
  get(accumulate_tilde_phi) = 0.;

  // Cellwise means, used in computing the cons/char transformations
  const auto mean_tilde_d = Scalar<double>{mean_value(get(*tilde_d), mesh)};
  const auto mean_tilde_tau = Scalar<double>{mean_value(get(*tilde_tau), mesh)};
  const auto mean_tilde_s = [&tilde_s, &mesh]() noexcept {
    tnsr::i<double, 3> result{};
    for (size_t i = 0; i < 3; ++i) {
      result.get(i) = mean_value(tilde_s->get(i), mesh);
    }
    return result;
  }();
  const auto mean_tilde_b = [&tilde_b, &mesh]() noexcept {
    tnsr::I<double, 3> result{};
    for (size_t i = 0; i < 3; ++i) {
      result.get(i) = mean_value(tilde_b->get(i), mesh);
    }
    return result;
  }();
  const auto mean_tilde_phi = Scalar<double>{mean_value(get(*tilde_phi), mesh)};
  const auto mean_lapse = Scalar<double>{mean_value(get(lapse), mesh)};
  const auto mean_shift = [&shift, &mesh]() noexcept {
    tnsr::I<double, 3> result{};
    for (size_t i = 0; i < 3; ++i) {
      result.get(i) = mean_value(shift.get(i), mesh);
    }
    return result;
  }();
  const auto mean_spatial_metric = [&spatial_metric, &mesh]() noexcept {
    tnsr::ii<double, 3> result{};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        result.get(i, j) = mean_value(spatial_metric.get(i, j), mesh);
      }
    }
    return result;
  }();

  bool some_component_was_limited = false;

  // Loop over directions, then compute chars w.r.t. this direction and limit
  for (size_t d = 0; d < 3; ++d) {
    const auto unit_vector = [&d]() noexcept {
      auto components = make_array<3>(0.);
      components[d] = 1.;
      return tnsr::i<double, 3>(components);
    }();
    const auto right_and_left =
        grmhd::ValenciaDivClean::Limiters::right_and_left_eigenvectors(
            mean_tilde_d, mean_tilde_tau, mean_tilde_s, mean_tilde_b,
            mean_tilde_phi, mean_lapse, mean_shift, mean_spatial_metric,
            equation_of_state, unit_vector);
    const auto& right = right_and_left.first;
    const auto& left = right_and_left.second;

    // Transform tensors to characteristics
    grmhd::ValenciaDivClean::Limiters::characteristic_fields(
        make_not_null(&char_v_div_clean_minus), make_not_null(&char_v_minus),
        make_not_null(&char_v_momentum), make_not_null(&char_v_plus),
        make_not_null(&char_v_div_clean_plus), *tilde_d, *tilde_tau, *tilde_s,
        *tilde_b, *tilde_phi, left);

    // Transform neighbor data and apply limiter
    const bool some_component_was_limited_with_this_unit_vector =
        prepare_and_apply_limiter(make_not_null(&char_v_div_clean_minus),
                                  make_not_null(&char_v_minus),
                                  make_not_null(&char_v_momentum),
                                  make_not_null(&char_v_plus),
                                  make_not_null(&char_v_div_clean_plus), left);

    some_component_was_limited =
        some_component_was_limited_with_this_unit_vector or
        some_component_was_limited;

    // Transform back to conserved variables. But skip the transformation if no
    // limiting occured with this unit vector.
    if (some_component_was_limited_with_this_unit_vector) {
      grmhd::ValenciaDivClean::Limiters::
          conserved_fields_from_characteristic_fields(
              make_not_null(&temp_tilde_d), make_not_null(&temp_tilde_tau),
              make_not_null(&temp_tilde_s), make_not_null(&temp_tilde_b),
              make_not_null(&temp_tilde_phi), char_v_div_clean_minus,
              char_v_minus, char_v_momentum, char_v_plus, char_v_div_clean_plus,
              right);
    } else {
      temp_tilde_d = *tilde_d;
      temp_tilde_tau = *tilde_tau;
      temp_tilde_s = *tilde_s;
      temp_tilde_b = *tilde_b;
      temp_tilde_phi = *tilde_phi;
    }

    // Add to running sum for averaging
    const double one_over_dim = 1.0 / static_cast<double>(3);
    get(accumulate_tilde_d) += one_over_dim * get(temp_tilde_d);
    get(accumulate_tilde_tau) += one_over_dim * get(temp_tilde_tau);
    for (size_t i = 0; i < 3; ++i) {
      accumulate_tilde_s.get(i) += one_over_dim * temp_tilde_s.get(i);
      accumulate_tilde_b.get(i) += one_over_dim * temp_tilde_b.get(i);
    }
    get(accumulate_tilde_phi) += one_over_dim * get(temp_tilde_phi);
  }  // for loop over dimensions

  *tilde_d = accumulate_tilde_d;
  *tilde_tau = accumulate_tilde_tau;
  *tilde_s = accumulate_tilde_s;
  *tilde_b = accumulate_tilde_b;
  *tilde_phi = accumulate_tilde_phi;
  return some_component_was_limited;
}

}  // namespace grmhd::ValenciaDivClean::Limiters
