// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"               // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd::ValenciaDivClean::Limiters {

template <size_t ThermodynamicDim>
std::pair<Matrix, Matrix> right_and_left_eigenvectors(
    const Scalar<double>& /*mean_tilde_d*/,
    const Scalar<double>& /*mean_tilde_tau*/,
    const tnsr::i<double, 3>& /*mean_tilde_s*/,
    const tnsr::I<double, 3>& /*mean_tilde_b*/,
    const Scalar<double>& /*mean_tilde_phi*/, const Scalar<double>& mean_lapse,
    const tnsr::I<double, 3>& mean_shift,
    const tnsr::ii<double, 3>& mean_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const tnsr::i<double, 3>& unit_vector) noexcept {
  // Compute fluid primitives from mean conserved state
  // TODO: do this using some code like PrimitiveFromConservative, but that
  // works for doubles. Will need to uncomment the args above.
  const Scalar<double> rest_mass_density{{{0.}}};
  const Scalar<double> specific_internal_energy{{{0.}}};
  const Scalar<double> specific_enthalpy{{{1.}}};
  const tnsr::I<double, 3> spatial_velocity{{{0.}}};
  const Scalar<double> lorentz_factor{{{1.}}};
  const tnsr::I<double, 3> magnetic_field{{{0.}}};

  return numerical_eigensystem(
             rest_mass_density, specific_internal_energy, specific_enthalpy,
             spatial_velocity, lorentz_factor, magnetic_field, mean_lapse,
             mean_shift, mean_spatial_metric, unit_vector, equation_of_state)
      .second;
}

void characteristic_fields(
    const gsl::not_null<tuples::TaggedTuple<
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
    const Matrix& left) noexcept {
  auto& char_v_div_clean_minus =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>>(
          *char_means);
  auto& char_v_minus =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::VMinus>>(*char_means);
  auto& char_v_momentum =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::VMomentum>>(*char_means);
  auto& char_v_plus =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::VPlus>>(*char_means);
  auto& char_v_div_clean_plus =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>(
          *char_means);

  const auto& tilde_d =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeD>>(cons_means);
  const auto& tilde_tau =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeTau>>(cons_means);
  const auto& tilde_s =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeS<>>>(cons_means);
  const auto& tilde_b =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildeB<>>>(cons_means);
  const auto& tilde_phi =
      get<::Tags::Mean<grmhd::ValenciaDivClean::Tags::TildePhi>>(cons_means);

  get(char_v_div_clean_minus) = left(0, 0) * get(tilde_d);
  get(char_v_minus) = left(1, 0) * get(tilde_d);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum.get(j) = left(j + 2, 0) * get(tilde_d);
  }
  get(char_v_plus) = left(7, 0) * get(tilde_tau);
  get(char_v_div_clean_plus) = left(8, 0) * get(tilde_tau);

  get(char_v_div_clean_minus) += left(0, 1) * get(tilde_tau);
  get(char_v_minus) += left(1, 1) * get(tilde_tau);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum.get(j) += left(j + 2, 1) * get(tilde_tau);
  }
  get(char_v_plus) += left(7, 1) * get(tilde_tau);
  get(char_v_div_clean_plus) += left(8, 1) * get(tilde_tau);

  for (size_t i = 0; i < 3; ++i) {
    get(char_v_div_clean_minus) += left(0, i + 2) * tilde_s.get(i);
    get(char_v_minus) += left(1, i + 2) * tilde_s.get(i);
    for (size_t j = 0; j < 5; ++j) {
      char_v_momentum.get(j) += left(j + 2, i + 2) * tilde_s.get(i);
    }
    get(char_v_plus) += left(7, i + 2) * tilde_s.get(i);
    get(char_v_div_clean_plus) += left(8, i + 2) * tilde_s.get(i);

    get(char_v_div_clean_minus) += left(0, i + 5) * tilde_b.get(i);
    get(char_v_minus) += left(1, i + 5) * tilde_b.get(i);
    for (size_t j = 0; j < 5; ++j) {
      char_v_momentum.get(j) += left(j + 2, i + 5) * tilde_b.get(i);
    }
    get(char_v_plus) += left(7, i + 5) * tilde_b.get(i);
    get(char_v_div_clean_plus) += left(8, i + 5) * tilde_b.get(i);
  }

  get(char_v_div_clean_minus) += left(0, 8) * get(tilde_phi);
  get(char_v_minus) += left(1, 8) * get(tilde_phi);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum.get(j) += left(j + 2, 8) * get(tilde_phi);
  }
  get(char_v_plus) += left(7, 8) * get(tilde_phi);
  get(char_v_div_clean_plus) += left(8, 8) * get(tilde_phi);
}

void characteristic_fields(
    const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
    const gsl::not_null<Scalar<DataVector>*> char_v_minus,
    const gsl::not_null<tnsr::I<DataVector, 3>*> char_v_momentum,
    const gsl::not_null<Scalar<DataVector>*> char_v_plus,
    const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3>& tilde_s,
    const tnsr::I<DataVector, 3>& tilde_b, const Scalar<DataVector>& tilde_phi,
    const Matrix& left) noexcept {
  get(*char_v_div_clean_minus) = left(0, 0) * get(tilde_d);
  get(*char_v_minus) = left(1, 0) * get(tilde_d);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum->get(j) = left(j + 2, 0) * get(tilde_d);
  }
  get(*char_v_plus) = left(7, 0) * get(tilde_tau);
  get(*char_v_div_clean_plus) = left(8, 0) * get(tilde_tau);

  get(*char_v_div_clean_minus) += left(0, 1) * get(tilde_tau);
  get(*char_v_minus) += left(1, 1) * get(tilde_tau);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum->get(j) += left(j + 2, 1) * get(tilde_tau);
  }
  get(*char_v_plus) += left(7, 1) * get(tilde_tau);
  get(*char_v_div_clean_plus) += left(8, 1) * get(tilde_tau);

  for (size_t i = 0; i < 3; ++i) {
    get(*char_v_div_clean_minus) += left(0, i + 2) * tilde_s.get(i);
    get(*char_v_minus) += left(1, i + 2) * tilde_s.get(i);
    for (size_t j = 0; j < 5; ++j) {
      char_v_momentum->get(j) += left(j + 2, i + 2) * tilde_s.get(i);
    }
    get(*char_v_plus) += left(7, i + 2) * tilde_s.get(i);
    get(*char_v_div_clean_plus) += left(8, i + 2) * tilde_s.get(i);

    get(*char_v_div_clean_minus) += left(0, i + 5) * tilde_b.get(i);
    get(*char_v_minus) += left(1, i + 5) * tilde_b.get(i);
    for (size_t j = 0; j < 5; ++j) {
      char_v_momentum->get(j) += left(j + 2, i + 5) * tilde_b.get(i);
    }
    get(*char_v_plus) += left(7, i + 5) * tilde_b.get(i);
    get(*char_v_div_clean_plus) += left(8, i + 5) * tilde_b.get(i);
  }

  get(*char_v_div_clean_minus) += left(0, 8) * get(tilde_phi);
  get(*char_v_minus) += left(1, 8) * get(tilde_phi);
  for (size_t j = 0; j < 5; ++j) {
    char_v_momentum->get(j) += left(j + 2, 8) * get(tilde_phi);
  }
  get(*char_v_plus) += left(7, 8) * get(tilde_phi);
  get(*char_v_div_clean_plus) += left(8, 8) * get(tilde_phi);
}

void characteristic_fields(
    const gsl::not_null<
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
    const Matrix& left) noexcept {
  characteristic_fields(
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>(*char_vars)),
      make_not_null(&get<grmhd::ValenciaDivClean::Tags::VMinus>(*char_vars)),
      make_not_null(&get<grmhd::ValenciaDivClean::Tags::VMomentum>(*char_vars)),
      make_not_null(&get<grmhd::ValenciaDivClean::Tags::VPlus>(*char_vars)),
      make_not_null(
          &get<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>(*char_vars)),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(cons_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(cons_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeS<>>(cons_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(cons_vars),
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(cons_vars), left);
}

void conserved_fields_from_characteristic_fields(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,
    const Scalar<DataVector>& char_v_div_clean_minus,
    const Scalar<DataVector>& char_v_minus,
    const tnsr::I<DataVector, 5>& char_v_momentum,
    const Scalar<DataVector>& char_v_plus,
    const Scalar<DataVector>& char_v_div_clean_plus,
    const Matrix& right) noexcept {
  get(*tilde_d) = right(0, 0) * get(char_v_div_clean_minus);
  get(*tilde_tau) = right(1, 0) * get(char_v_div_clean_minus);
  for (size_t j = 0; j < 3; ++j) {
    tilde_s->get(j) = right(j + 2, 0) * get(char_v_div_clean_minus);
    tilde_b->get(j) = right(j + 5, 0) * get(char_v_div_clean_minus);
  }
  get(*tilde_phi) = right(8, 0) * get(char_v_div_clean_minus);

  get(*tilde_d) += right(0, 1) * get(char_v_minus);
  get(*tilde_tau) += right(1, 1) * get(char_v_minus);
  for (size_t j = 0; j < 3; ++j) {
    tilde_s->get(j) += right(j + 2, 1) * get(char_v_minus);
    tilde_b->get(j) += right(j + 5, 1) * get(char_v_minus);
  }
  get(*tilde_phi) += right(8, 1) * get(char_v_minus);

  for (size_t i = 0; i < 5; ++i) {
    get(*tilde_d) += right(0, i + 2) * char_v_momentum.get(i);
    get(*tilde_tau) += right(1, i + 2) * char_v_momentum.get(i);
    for (size_t j = 0; j < 3; ++j) {
      tilde_s->get(j) += right(j + 2, i + 2) * char_v_momentum.get(i);
      tilde_b->get(j) += right(j + 5, i + 2) * char_v_momentum.get(i);
    }
    get(*tilde_phi) += right(8, i + 2) * char_v_momentum.get(i);
  }

  get(*tilde_d) += right(0, 7) * get(char_v_plus);
  get(*tilde_tau) += right(1, 7) * get(char_v_plus);
  for (size_t j = 0; j < 3; ++j) {
    tilde_s->get(j) += right(j + 2, 7) * get(char_v_plus);
    tilde_b->get(j) += right(j + 5, 7) * get(char_v_plus);
  }
  get(*tilde_phi) += right(8, 7) * get(char_v_plus);

  get(*tilde_d) += right(0, 8) * get(char_v_div_clean_plus);
  get(*tilde_tau) += right(1, 8) * get(char_v_div_clean_plus);
  for (size_t j = 0; j < 3; ++j) {
    tilde_s->get(j) += right(j + 2, 8) * get(char_v_div_clean_plus);
    tilde_b->get(j) += right(j + 5, 8) * get(char_v_div_clean_plus);
  }
  get(*tilde_phi) += right(8, 8) * get(char_v_div_clean_plus);
}

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::pair<Matrix, Matrix> right_and_left_eigenvectors(              \
      const Scalar<double>&, const Scalar<double>&, const tnsr::i<double, 3>&, \
      const tnsr::I<double, 3>&, const Scalar<double>&, const Scalar<double>&, \
      const tnsr::I<double, 3>&, const tnsr::ii<double, 3>&,                   \
      const EquationsOfState::EquationOfState<true, THERMODIM(data)>&,         \
      const tnsr::i<double, 3>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2))

#undef THERMODIM
#undef INSTANTIATE

}  // namespace grmhd::ValenciaDivClean::Limiters
