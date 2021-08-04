// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

// This computes a unit vector. It is NOT uniformly distributed in angle,
// but for this test the angular distribution is not important.
template <size_t Dim>
tnsr::i<double, Dim> random_unit_vector(
    const gsl::not_null<std::mt19937*>& gen,
    const gsl::not_null<std::uniform_real_distribution<>*>& dist) noexcept {
  const double used_for_size = 0.;
  auto result =
      make_with_random_values<tnsr::i<double, Dim>>(gen, dist, used_for_size);
  double vector_magnitude = get(magnitude(result));
  // Though highly unlikely, the vector could have length nearly 0. If this
  // happens, we edit the vector to make it non-zero.
  if (vector_magnitude < 1e-3) {
    get<0>(result) += 0.9;
    vector_magnitude = get(magnitude(result));
  }
  // Make unit magnitude
  for (auto& n_i : result) {
    n_i /= vector_magnitude;
  }
  return result;
}

void prepare_data_for_test(
    const gsl::not_null<Scalar<DataVector>*>& tilde_d,
    const gsl::not_null<Scalar<DataVector>*>& tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3>*>& tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3>*>& tilde_b,
    const gsl::not_null<Scalar<DataVector>*>& tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>& lapse,
    const gsl::not_null<tnsr::I<DataVector, 3>*>& shift,
    const gsl::not_null<tnsr::ii<DataVector, 3>*>& spatial_metric,
    const gsl::not_null<Scalar<double>*>& mean_tilde_d,
    const gsl::not_null<Scalar<double>*>& mean_tilde_tau,
    const gsl::not_null<tnsr::i<double, 3>*>& mean_tilde_s,
    const gsl::not_null<tnsr::I<double, 3>*>& mean_tilde_b,
    const gsl::not_null<Scalar<double>*>& mean_tilde_phi,
    const gsl::not_null<Scalar<double>*>& mean_lapse,
    const gsl::not_null<tnsr::I<double, 3>*>& mean_shift,
    const gsl::not_null<tnsr::ii<double, 3>*>& mean_spatial_metric,
    const gsl::not_null<Matrix*>& right, const gsl::not_null<Matrix*>& left,
    const Mesh<3>& mesh,
    const EquationsOfState::IdealFluid<true>& equation_of_state,
    const bool generate_random_vector = true) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  std::uniform_real_distribution<> distribution_positive(1e-3, 1.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_distribution_positive = make_not_null(&distribution_positive);

  const auto unit_vector = [&generate_random_vector, &nn_generator,
                            &nn_distribution]() noexcept {
    if (generate_random_vector) {
      return random_unit_vector<3>(nn_generator, nn_distribution);
    } else {
      // Return unit vector pointing in the "last" dimension
      tnsr::i<double, 3> result{{0.}};
      get<2>(result) = 1.;
      return result;
    }
  }();

  const DataVector used_for_size(mesh.number_of_grid_points(), 0.);
  *lapse = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  get(*lapse) = 1. + 1e-3 * get(*lapse);
  *shift = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) *= 1e-3;
  }
  *spatial_metric = make_with_random_values<tnsr::ii<DataVector, 3>>(
      nn_generator, nn_distribution_positive, used_for_size);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      spatial_metric->get(i, j) *= 1e-3;
    }
    spatial_metric->get(i, i) += 1.;
  }
  auto sqrt_det_spatial_metric = determinant(*spatial_metric);
  get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

  // Derive all fluid quantities from the primitive variables
  const auto rest_mass_density = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution_positive, used_for_size);
  const auto specific_internal_energy =
      make_with_random_values<Scalar<DataVector>>(
          nn_generator, nn_distribution_positive, used_for_size);
  const auto spatial_velocity = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  const auto magnetic_field = make_with_random_values<tnsr::I<DataVector, 3>>(
      nn_generator, nn_distribution, used_for_size);
  auto divergence_cleaning_field = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution_positive, used_for_size);

  const auto pressure = equation_of_state.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto lorentz_factor = hydro::lorentz_factor(
      dot_product(spatial_velocity, spatial_velocity, *spatial_metric));

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, rest_mass_density,
      specific_internal_energy, specific_enthalpy, pressure, spatial_velocity,
      lorentz_factor, magnetic_field, sqrt_det_spatial_metric, *spatial_metric,
      divergence_cleaning_field);

  get(*mean_tilde_d) = mean_value(get(*tilde_d), mesh);
  get(*mean_tilde_tau) = mean_value(get(*tilde_tau), mesh);
  for (size_t i = 0; i < 3; ++i) {
    mean_tilde_s->get(i) = mean_value(tilde_s->get(i), mesh);
    mean_tilde_b->get(i) = mean_value(tilde_b->get(i), mesh);
  }
  get(*mean_tilde_phi) = mean_value(get(*tilde_phi), mesh);
  get(*mean_lapse) = mean_value(get(*lapse), mesh);
  for (size_t i = 0; i < 3; ++i) {
    mean_shift->get(i) = mean_value(shift->get(i), mesh);
    for (size_t j = i; j < 3; ++j) {
      mean_spatial_metric->get(i, j) =
          mean_value(spatial_metric->get(i, j), mesh);
    }
  }

  const auto right_and_left =
      grmhd::ValenciaDivClean::Limiters::right_and_left_eigenvectors(
          *mean_tilde_d, *mean_tilde_tau, *mean_tilde_s, *mean_tilde_b,
          *mean_tilde_phi, *mean_lapse, *mean_shift, *mean_spatial_metric,
          equation_of_state, unit_vector);
  *right = right_and_left.first;
  *left = right_and_left.second;
}

void test_characteristic_helpers() noexcept {
  INFO("Testing characteristic helpers");
  const Mesh<3> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<true> equation_of_state{5. / 3.};

  Scalar<DataVector> tilde_d;
  Scalar<DataVector> tilde_tau;
  tnsr::i<DataVector, 3> tilde_s;
  tnsr::I<DataVector, 3> tilde_b;
  Scalar<DataVector> tilde_phi;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::ii<DataVector, 3> spatial_metric;
  Scalar<double> mean_tilde_d;
  Scalar<double> mean_tilde_tau;
  tnsr::i<double, 3> mean_tilde_s;
  tnsr::I<double, 3> mean_tilde_b;
  Scalar<double> mean_tilde_phi;
  Scalar<double> mean_lapse;
  tnsr::I<double, 3> mean_shift;
  tnsr::ii<double, 3> mean_spatial_metric;
  Matrix right;
  Matrix left;

  prepare_data_for_test(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), make_not_null(&lapse), make_not_null(&shift),
      make_not_null(&spatial_metric), make_not_null(&mean_tilde_d),
      make_not_null(&mean_tilde_tau), make_not_null(&mean_tilde_s),
      make_not_null(&mean_tilde_b), make_not_null(&mean_tilde_phi),
      make_not_null(&mean_lapse), make_not_null(&mean_shift),
      make_not_null(&mean_spatial_metric), make_not_null(&right),
      make_not_null(&left), mesh, equation_of_state);

  // First test that the tensor-transform helpers are inverses of each
  // other. This is a sanity check of the two tensor-transform helpers and
  // of right_and_left_eigenvectors.
  Scalar<DataVector> v_div_clean_minus{};
  Scalar<DataVector> v_minus{};
  tnsr::I<DataVector, 5> v_momentum{};
  Scalar<DataVector> v_plus{};
  Scalar<DataVector> v_div_clean_plus{};
  grmhd::ValenciaDivClean::Limiters::characteristic_fields(
      make_not_null(&v_div_clean_minus), make_not_null(&v_minus),
      make_not_null(&v_momentum), make_not_null(&v_plus),
      make_not_null(&v_div_clean_plus), tilde_d, tilde_tau, tilde_s, tilde_b,
      tilde_phi, left);
  Scalar<DataVector> recovered_tilde_d{};
  Scalar<DataVector> recovered_tilde_tau{};
  tnsr::i<DataVector, 3> recovered_tilde_s{};
  tnsr::I<DataVector, 3> recovered_tilde_b{};
  Scalar<DataVector> recovered_tilde_phi{};
  grmhd::ValenciaDivClean::Limiters::
      conserved_fields_from_characteristic_fields(
          make_not_null(&recovered_tilde_d),
          make_not_null(&recovered_tilde_tau),
          make_not_null(&recovered_tilde_s), make_not_null(&recovered_tilde_b),
          make_not_null(&recovered_tilde_phi), v_div_clean_minus, v_minus,
          v_momentum, v_plus, v_div_clean_plus, right);
  CHECK_ITERABLE_APPROX(tilde_d, recovered_tilde_d);
  CHECK_ITERABLE_APPROX(tilde_tau, recovered_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_s, recovered_tilde_s);
  CHECK_ITERABLE_APPROX(tilde_b, recovered_tilde_b);
  CHECK_ITERABLE_APPROX(tilde_phi, recovered_tilde_phi);

  // Test that the cell-average helper matches the tensor helper
  tuples::TaggedTuple<Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>,
                      Tags::Mean<grmhd::ValenciaDivClean::Tags::VMinus>,
                      Tags::Mean<grmhd::ValenciaDivClean::Tags::VMomentum>,
                      Tags::Mean<grmhd::ValenciaDivClean::Tags::VPlus>,
                      Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>
      mean_char_vars;
  grmhd::ValenciaDivClean::Limiters::characteristic_fields(
      make_not_null(&mean_char_vars),
      {mean_tilde_d, mean_tilde_tau, mean_tilde_s, mean_tilde_b,
       mean_tilde_phi},
      left);
  CHECK(get(get<Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>>(
            mean_char_vars)) ==
        approx(mean_value(get(v_div_clean_minus), mesh)));
  CHECK(get(get<Tags::Mean<grmhd::ValenciaDivClean::Tags::VMinus>>(
            mean_char_vars)) == approx(mean_value(get(v_minus), mesh)));
  for (size_t i = 0; i < 5; ++i) {
    CHECK(get<Tags::Mean<grmhd::ValenciaDivClean::Tags::VMomentum>>(
              mean_char_vars)
              .get(i) == approx(mean_value(v_momentum.get(i), mesh)));
  }
  CHECK(get(get<Tags::Mean<grmhd::ValenciaDivClean::Tags::VPlus>>(
            mean_char_vars)) == approx(mean_value(get(v_plus), mesh)));
  CHECK(get(get<Tags::Mean<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>(
            mean_char_vars)) ==
        approx(mean_value(get(v_div_clean_plus), mesh)));

  // Test that the Variables helper matches the tensor helper
  Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::VDivCleanMinus,
                       grmhd::ValenciaDivClean::Tags::VMinus,
                       grmhd::ValenciaDivClean::Tags::VMomentum,
                       grmhd::ValenciaDivClean::Tags::VPlus,
                       grmhd::ValenciaDivClean::Tags::VDivCleanPlus>>
      char_vars(mesh.number_of_grid_points());
  Variables<tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                       grmhd::ValenciaDivClean::Tags::TildeTau,
                       grmhd::ValenciaDivClean::Tags::TildeS<>,
                       grmhd::ValenciaDivClean::Tags::TildeB<>,
                       grmhd::ValenciaDivClean::Tags::TildePhi>>
      cons_vars(mesh.number_of_grid_points());
  get<grmhd::ValenciaDivClean::Tags::TildeD>(cons_vars) = tilde_d;
  get<grmhd::ValenciaDivClean::Tags::TildeTau>(cons_vars) = tilde_tau;
  get<grmhd::ValenciaDivClean::Tags::TildeS<>>(cons_vars) = tilde_s;
  get<grmhd::ValenciaDivClean::Tags::TildeB<>>(cons_vars) = tilde_b;
  get<grmhd::ValenciaDivClean::Tags::TildePhi>(cons_vars) = tilde_phi;
  grmhd::ValenciaDivClean::Limiters::characteristic_fields(
      make_not_null(&char_vars), cons_vars, left);
  CHECK_ITERABLE_APPROX(
      get<grmhd::ValenciaDivClean::Tags::VDivCleanMinus>(char_vars),
      v_div_clean_minus);
  CHECK_ITERABLE_APPROX(get<grmhd::ValenciaDivClean::Tags::VMinus>(char_vars),
                        v_minus);
  CHECK_ITERABLE_APPROX(
      get<grmhd::ValenciaDivClean::Tags::VMomentum>(char_vars), v_momentum);
  CHECK_ITERABLE_APPROX(get<grmhd::ValenciaDivClean::Tags::VPlus>(char_vars),
                        v_plus);
  CHECK_ITERABLE_APPROX(
      get<grmhd::ValenciaDivClean::Tags::VDivCleanPlus>(char_vars),
      v_div_clean_plus);
}

void test_apply_limiter_to_char_fields() noexcept {
  INFO("Testing apply_limiter_to_characteristic_fields_in_all_directions");
  const Mesh<3> mesh(3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const EquationsOfState::IdealFluid<true> equation_of_state{5. / 3.};

  Scalar<DataVector> tilde_d;
  Scalar<DataVector> tilde_tau;
  tnsr::i<DataVector, 3> tilde_s;
  tnsr::I<DataVector, 3> tilde_b;
  Scalar<DataVector> tilde_phi;
  Scalar<DataVector> lapse;
  tnsr::I<DataVector, 3> shift;
  tnsr::ii<DataVector, 3> spatial_metric;
  Scalar<double> mean_tilde_d;
  Scalar<double> mean_tilde_tau;
  tnsr::i<double, 3> mean_tilde_s;
  tnsr::I<double, 3> mean_tilde_b;
  Scalar<double> mean_tilde_phi;
  Scalar<double> mean_lapse;
  tnsr::I<double, 3> mean_shift;
  tnsr::ii<double, 3> mean_spatial_metric;
  Matrix right;
  Matrix left;

  const bool generate_random_vector = false;
  prepare_data_for_test(
      make_not_null(&tilde_d), make_not_null(&tilde_tau),
      make_not_null(&tilde_s), make_not_null(&tilde_b),
      make_not_null(&tilde_phi), make_not_null(&lapse), make_not_null(&shift),
      make_not_null(&spatial_metric), make_not_null(&mean_tilde_d),
      make_not_null(&mean_tilde_tau), make_not_null(&mean_tilde_s),
      make_not_null(&mean_tilde_b), make_not_null(&mean_tilde_phi),
      make_not_null(&mean_lapse), make_not_null(&mean_shift),
      make_not_null(&mean_spatial_metric), make_not_null(&right),
      make_not_null(&left), mesh, equation_of_state, generate_random_vector);

  // Check case where limiter does nothing
  const auto noop_expected_tilde_d = tilde_d;
  const auto noop_expected_tilde_tau = tilde_tau;
  const auto noop_expected_tilde_s = tilde_s;
  const auto noop_expected_tilde_b = tilde_b;
  const auto noop_expected_tilde_phi = tilde_phi;
  const auto noop_limiter_wrapper =
      [](const gsl::not_null<Scalar<DataVector>*> /*char_v_div_clean_minus*/,
         const gsl::not_null<Scalar<DataVector>*> /*char_v_minus*/,
         const gsl::not_null<tnsr::I<DataVector, 5>*> /*char_v_momentum*/,
         const gsl::not_null<Scalar<DataVector>*> /*char_v_plus*/,
         const gsl::not_null<Scalar<DataVector>*> /*char_v_div_clean_plus*/,
         const Matrix& /*left*/) noexcept -> bool { return false; };

  const bool noop_result = grmhd::ValenciaDivClean::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          make_not_null(&tilde_d), make_not_null(&tilde_tau),
          make_not_null(&tilde_s), make_not_null(&tilde_b),
          make_not_null(&tilde_phi), lapse, shift, spatial_metric, mesh,
          equation_of_state, noop_limiter_wrapper);
  CHECK_FALSE(noop_result);
  CHECK_ITERABLE_APPROX(tilde_d, noop_expected_tilde_d);
  CHECK_ITERABLE_APPROX(tilde_tau, noop_expected_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_s, noop_expected_tilde_s);
  CHECK_ITERABLE_APPROX(tilde_b, noop_expected_tilde_b);
  CHECK_ITERABLE_APPROX(tilde_phi, noop_expected_tilde_phi);

  // Check with a silly (and completely nonphysical) limiter that scales char
  // fields, which is relatively easy to check. The solution is set to 0 on the
  // first dims (d < 2), and scaled by some integers on dim d == 2
  size_t current_dim = 0;
  const auto silly_limiter_wrapper =
      [&current_dim](
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_minus,
          const gsl::not_null<Scalar<DataVector>*> char_v_minus,
          const gsl::not_null<tnsr::I<DataVector, 5>*> char_v_momentum,
          const gsl::not_null<Scalar<DataVector>*> char_v_plus,
          const gsl::not_null<Scalar<DataVector>*> char_v_div_clean_plus,
          const Matrix& /*left*/) noexcept -> bool {
    if (current_dim == 2) {
      get(*char_v_div_clean_minus) *= 2.;
      // no change to char_v_minus, char_v_momentum, char_v_plus
      get(*char_v_div_clean_plus) = 0.;
    } else {
      get(*char_v_div_clean_minus) = 0.;
      get(*char_v_minus) = 0.;
      for (size_t i = 0; i < 5; ++i) {
        char_v_momentum->get(i) = 0.;
      }
      get(*char_v_plus) = 0.;
      get(*char_v_div_clean_plus) = 0.;
      current_dim++;
    }
    return true;
  };

  Matrix silly_limiter_action(8, 8, 0.);
  silly_limiter_action(0, 0) = 2. / static_cast<double>(3);
  silly_limiter_action(1, 1) = 1. / static_cast<double>(3);
  for (size_t i = 0; i < 5; ++i) {
    silly_limiter_action(i + 2, i + 2) = 1. / static_cast<double>(3);
  }
  silly_limiter_action(7, 7) = 1. / static_cast<double>(3);
  silly_limiter_action(8, 8) = 0.;
  const Matrix silly = right * silly_limiter_action * left;

  auto expected_tilde_d = tilde_d;
  auto expected_tilde_tau = tilde_tau;
  auto expected_tilde_s = tilde_s;
  auto expected_tilde_b = tilde_b;
  auto expected_tilde_phi = tilde_phi;

  get(expected_tilde_d) = silly(0, 0) * get(tilde_d);
  get(expected_tilde_tau) = silly(1, 0) * get(tilde_d);
  for (size_t j = 0; j < 3; ++j) {
    expected_tilde_s.get(j) = silly(j + 2, 0) * get(tilde_d);
    expected_tilde_b.get(j) = silly(j + 5, 0) * get(tilde_d);
  }
  get(expected_tilde_phi) = silly(8, 0) * get(tilde_d);

  get(expected_tilde_d) += silly(0, 1) * get(tilde_tau);
  get(expected_tilde_tau) += silly(1, 1) * get(tilde_tau);
  for (size_t j = 0; j < 3; ++j) {
    expected_tilde_s.get(j) += silly(j + 2, 1) * get(tilde_tau);
    expected_tilde_b.get(j) += silly(j + 5, 1) * get(tilde_tau);
  }
  get(expected_tilde_phi) += silly(8, 1) * get(tilde_tau);

  for (size_t i = 0; i < 3; ++i) {
    get(expected_tilde_d) += silly(0, i + 2) * tilde_s.get(i);
    get(expected_tilde_tau) += silly(1, i + 2) * tilde_s.get(i);
    for (size_t j = 0; j < 3; ++j) {
      expected_tilde_s.get(j) += silly(j + 2, i + 2) * tilde_s.get(i);
      expected_tilde_b.get(j) += silly(j + 5, i + 2) * tilde_s.get(i);
    }
    get(expected_tilde_phi) += silly(8, i + 2) * tilde_s.get(i);

    get(expected_tilde_d) += silly(0, i + 5) * tilde_b.get(i);
    get(expected_tilde_tau) += silly(1, i + 5) * tilde_b.get(i);
    for (size_t j = 0; j < 3; ++j) {
      expected_tilde_s.get(j) += silly(j + 2, i + 5) * tilde_b.get(i);
      expected_tilde_b.get(j) += silly(j + 5, i + 5) * tilde_b.get(i);
    }
    get(expected_tilde_phi) += silly(8, i + 5) * tilde_b.get(i);
  }

  get(expected_tilde_d) += silly(0, 8) * get(tilde_phi);
  get(expected_tilde_tau) += silly(1, 8) * get(tilde_phi);
  for (size_t j = 0; j < 3; ++j) {
    expected_tilde_s.get(j) += silly(j + 2, 8) * get(tilde_phi);
    expected_tilde_b.get(j) += silly(j + 5, 8) * get(tilde_phi);
  }
  get(expected_tilde_phi) += silly(8, 8) * get(tilde_phi);

  const bool silly_result = grmhd::ValenciaDivClean::Limiters::
      apply_limiter_to_characteristic_fields_in_all_directions(
          make_not_null(&tilde_d), make_not_null(&tilde_tau),
          make_not_null(&tilde_s), make_not_null(&tilde_b),
          make_not_null(&tilde_phi), lapse, shift, spatial_metric, mesh,
          equation_of_state, silly_limiter_wrapper);
  CHECK(silly_result);
  CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
  CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
  CHECK_ITERABLE_APPROX(tilde_b, expected_tilde_b);
  CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.CharacteristicHelpers",
                  "[Unit][Evolution]") {
  test_characteristic_helpers();
  test_apply_limiter_to_char_fields();
}
