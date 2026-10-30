// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Tags.hpp"
#include "PointwiseFunctions/Elasticity/PotentialEnergy.hpp"
#include "PointwiseFunctions/Elasticity/Stress.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// The executables observe the bulk and shear densities through this alias:
// both compute tags in 3D, none in 2D (see the alias for why)
static_assert(tmpl::list_contains_v<
              Elasticity::Tags::bulk_shear_observables<3>,
              Elasticity::Tags::BulkPotentialEnergyDensityCompute<3>>);
static_assert(tmpl::list_contains_v<
              Elasticity::Tags::bulk_shear_observables<3>,
              Elasticity::Tags::ShearPotentialEnergyDensityCompute<3>>);
static_assert(
    std::is_same_v<Elasticity::Tags::bulk_shear_observables<2>, tmpl::list<>>);

template <size_t Dim>
Scalar<DataVector> potential_energy_density(
    // The wrapper constructs a constitutive relation for use in
    // check_with_random_values.
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& strain,
    const tnsr::I<DataVector, Dim>& coordinates, const double bulk_modulus,
    const double shear_modulus) {
  Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  return Elasticity::potential_energy_density<Dim>(
      strain, coordinates, std::move(constitutive_relation));
}

template <size_t Dim>
void test_elastic_potential_energy(const DataVector& used_for_size) {
  pypp::check_with_random_values<4>(
      &potential_energy_density<Dim>, "Elasticity.PotentialEnergy",
      "potential_energy_density", {{{-1., 1.}, {0., 1.}, {0., 1.}, {0., 1.}}},
      used_for_size);
}

template <size_t Dim>
void test_compute_tags(const DataVector& used_for_size) {
  CAPTURE(Dim);
  using strain_tag = Elasticity::Tags::Strain<Dim>;
  using energy_tag = Elasticity::Tags::PotentialEnergyDensity<Dim>;
  using constitutive_relation_tag = Elasticity::Tags::ConstitutiveRelation<Dim>;
  using coordinates_tag = domain::Tags::Coordinates<Dim, Frame::Inertial>;
  TestHelpers::db::test_compute_tag<Elasticity::Tags::StressCompute<Dim>>(
      "Stress");
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::PotentialEnergyDensityCompute<Dim>>(
      "PotentialEnergyDensity");
  const size_t num_points = used_for_size.size();
  {
    INFO("Energy");
    std::unique_ptr<
        Elasticity::ConstitutiveRelations::ConstitutiveRelation<Dim>>
        constitutive_relation = std::make_unique<
            Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>>(3.,
                                                                          4.);
    const auto box =
        db::create<db::AddSimpleTags<strain_tag, constitutive_relation_tag,
                                     coordinates_tag>,
                   db::AddComputeTags<
                       Elasticity::Tags::StressCompute<Dim>,
                       Elasticity::Tags::PotentialEnergyDensityCompute<Dim>>>(
            tnsr::ii<DataVector, Dim>{num_points, 1.},
            std::move(constitutive_relation),
            tnsr::I<DataVector, Dim>{num_points, 2.});

    const auto expected_energy = Elasticity::potential_energy_density<Dim>(
        get<strain_tag>(box), get<coordinates_tag>(box),
        get<constitutive_relation_tag>(box));
    CHECK(get<energy_tag>(box) == expected_energy);
  }
}

template <size_t Dim>
void test_bulk_shear_split(const DataVector& used_for_size) {
  CAPTURE(Dim);
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::BulkPotentialEnergyDensityCompute<Dim>>(
      "BulkPotentialEnergyDensity");
  TestHelpers::db::test_compute_tag<
      Elasticity::Tags::ShearPotentialEnergyDensityCompute<Dim>>(
      "ShearPotentialEnergyDensity");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const auto strain = make_with_random_values<tnsr::ii<DataVector, Dim>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, Dim>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const double bulk_modulus = 3.;
  const double shear_modulus = 4.;
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<Dim>
      constitutive_relation{bulk_modulus, shear_modulus};
  tnsr::II<DataVector, Dim> stress{used_for_size.size()};
  constitutive_relation.stress(make_not_null(&stress), strain, x);
  Scalar<DataVector> bulk_energy{used_for_size.size()};
  Scalar<DataVector> shear_energy{used_for_size.size()};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  {
    INFO("Bulk + shear = total");
    const auto total_energy = Elasticity::potential_energy_density<Dim>(
        strain, x, constitutive_relation);
    const DataVector bulk_plus_shear = get(bulk_energy) + get(shear_energy);
    CHECK_ITERABLE_APPROX(bulk_plus_shear, get(total_energy));
  }
  if constexpr (Dim == 3) {
    INFO("Isotropic material: U_bulk = K Theta^2 / 2, U_shear = mu e_ij e_ij");
    DataVector trace_strain = get<0, 0>(strain);
    for (size_t d = 1; d < Dim; ++d) {
      trace_strain += strain.get(d, d);
    }
    DataVector deviatoric_square{used_for_size.size(), 0.};
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        DataVector deviatoric_strain = strain.get(i, j);
        if (i == j) {
          deviatoric_strain -= trace_strain / static_cast<double>(Dim);
        }
        deviatoric_square += square(deviatoric_strain);
      }
    }
    const DataVector expected_bulk = 0.5 * bulk_modulus * square(trace_strain);
    const DataVector expected_shear = shear_modulus * deviatoric_square;
    CHECK_ITERABLE_APPROX(get(bulk_energy), expected_bulk);
    CHECK_ITERABLE_APPROX(get(shear_energy), expected_shear);
  }
}

// Random strain and random stress, drawn independently: the observers need no
// physical relation between the two, so the deviatoric parts are in general not
// coaxial. PotentialEnergy.py uses the volumetric and deviatoric projectors,
// -1/2 S_ij P_ijkl T^kl with P_ijkl = (d_ik d_jl + d_il d_jk) / 2
// - d_ij d_kl / Dim; it forms no deviator and has no factor 2 for the
// off-diagonal terms, unlike the C++.
template <size_t Dim>
void test_bulk_shear_random_values(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &Elasticity::bulk_potential_energy_density<Dim>,
      "Elasticity.PotentialEnergy", {"bulk_potential_energy_density"},
      {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &Elasticity::shear_potential_energy_density<Dim>,
      "Elasticity.PotentialEnergy", {"shear_potential_energy_density"},
      {{{-1., 1.}}}, used_for_size);
}

// Only the upper triangle is read (symmetric storage)
template <typename TensorType, size_t Dim>
TensorType symmetric_tensor(
    const std::array<std::array<double, Dim>, Dim>& components,
    const size_t num_points) {
  TensorType result{num_points};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result.get(i, j) = gsl::at(gsl::at(components, i), j);
    }
  }
  return result;
}

// Fixed strain S and stress T with every component nonzero. S and T were
// chosen independently of each other, not generated from one another by a
// constitutive relation, so their deviatoric parts are not coaxial. All
// entries are binary fractions and all reference values are exact (checked in
// rational arithmetic). Theta = S_ii, Sigma = T^ii, e = S - Theta / Dim delta,
// s = T - Sigma / Dim delta. Each shear value is derived by two routes,
// -1/2 e_ij s^ij and -1/2 S_ij T^ij + Theta Sigma / (2 Dim).
template <size_t Dim>
struct FixedStrainAndStress;

template <>
struct FixedStrainAndStress<2> {
  static constexpr std::array<std::array<double, 2>, 2> strain{
      {{1., 0.5}, {0.5, -3.}}};
  static constexpr std::array<std::array<double, 2>, 2> stress{
      {{2., -1.}, {-1., 4.}}};
  // Theta = -2, Sigma = 6: U_bulk = -Theta Sigma / 4 = 3 (with the 3D factor
  // 1/6 it would be 2)
  static constexpr double bulk = 3.;
  // e = S + delta has diagonal (2, -2), s = T - 3 delta has diagonal (-1, 1),
  // off-diagonal parts unchanged: e_ij s^ij = -2 - 2 + 2 (0.5 * -1) = -5, so
  // U_shear = 5/2. Other route: S_ij T^ij = 2 - 12 + 2 (-0.5) = -11, so
  // 11/2 - 3 = 5/2
  static constexpr double shear = 2.5;
};

template <>
struct FixedStrainAndStress<3> {
  static constexpr std::array<std::array<double, 3>, 3> strain{
      {{1., 0.5, -0.25}, {0.5, -2., 0.75}, {-0.25, 0.75, 4.}}};
  static constexpr std::array<std::array<double, 3>, 3> stress{
      {{-3., 1., 0.5}, {1., 2., -1.5}, {0.5, -1.5, -5.}}};
  // Theta = 1 - 2 + 4 = 3, Sigma = -3 + 2 - 5 = -6:
  // U_bulk = -Theta Sigma / 6 = 3
  static constexpr double bulk = 3.;
  // e = S - delta has diagonal (0, -3, 3), s = T + 2 delta has diagonal
  // (-1, 4, -3), off-diagonal parts unchanged: e_ij s^ij = 0 - 12 - 9
  // + 2 (1/2 - 1/8 - 9/8) = -45/2, so U_shear = 45/4. Other route:
  // S_ij T^ij = -3 - 4 - 20 + 2 (-3/4) = -57/2, so 57/4 - 3 = 45/4. With a
  // factor 1 instead of 2 on the off-diagonal terms it would be 87/8
  static constexpr double shear = 11.25;
};

template <size_t Dim>
void test_bulk_shear_fixed_values() {
  INFO("Bulk and shear densities for a fixed strain and stress");
  CAPTURE(Dim);
  const size_t num_points = 3;
  const auto strain = symmetric_tensor<tnsr::ii<DataVector, Dim>>(
      FixedStrainAndStress<Dim>::strain, num_points);
  const auto stress = symmetric_tensor<tnsr::II<DataVector, Dim>>(
      FixedStrainAndStress<Dim>::stress, num_points);
  Scalar<DataVector> bulk_energy{num_points};
  Scalar<DataVector> shear_energy{num_points};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  CHECK_ITERABLE_APPROX(
      get(bulk_energy),
      DataVector(num_points, FixedStrainAndStress<Dim>::bulk));
  CHECK_ITERABLE_APPROX(
      get(shear_energy),
      DataVector(num_points, FixedStrainAndStress<Dim>::shear));
}

// The compute tags evaluated through a DataBox, compared against the fixed
// values above, which do not come from the functions under test. A compute tag
// bound to the wrong function (bulk <-> shear) fails. Swapped arguments do not
// compile (strain is `ii`, stress is `II`); they would not change the values
// either, since both densities are symmetric in S and T.
template <size_t Dim>
void test_bulk_shear_compute_tag_values() {
  INFO("Bulk and shear compute tags in a DataBox");
  CAPTURE(Dim);
  const size_t num_points = 3;
  const auto box = db::create<
      db::AddSimpleTags<Elasticity::Tags::Strain<Dim>,
                        Elasticity::Tags::Stress<Dim>>,
      db::AddComputeTags<
          Elasticity::Tags::BulkPotentialEnergyDensityCompute<Dim>,
          Elasticity::Tags::ShearPotentialEnergyDensityCompute<Dim>>>(
      symmetric_tensor<tnsr::ii<DataVector, Dim>>(
          FixedStrainAndStress<Dim>::strain, num_points),
      symmetric_tensor<tnsr::II<DataVector, Dim>>(
          FixedStrainAndStress<Dim>::stress, num_points));
  CHECK_ITERABLE_APPROX(
      get(db::get<Elasticity::Tags::BulkPotentialEnergyDensity<Dim>>(box)),
      DataVector(num_points, FixedStrainAndStress<Dim>::bulk));
  CHECK_ITERABLE_APPROX(
      get(db::get<Elasticity::Tags::ShearPotentialEnergyDensity<Dim>>(box)),
      DataVector(num_points, FixedStrainAndStress<Dim>::shear));
}

// Isotropic material, K = 3, mu = 4 (arbitrary units), stress
// T = -K Theta delta - 2 mu e. The stresses are written out by hand; that
// IsotropicHomogeneous gives the same stress checks the hand derivation, and
// the observers are fed the hand-written stress.
void check_isotropic_split(
    const std::array<std::array<double, 3>, 3>& strain_components,
    const std::array<std::array<double, 3>, 3>& stress_components,
    const double expected_bulk, const double expected_shear) {
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      constitutive_relation{3., 4.};
  const size_t num_points = 3;
  const auto x =
      make_with_value<tnsr::I<DataVector, 3>>(DataVector{num_points}, 0.);
  const auto strain =
      symmetric_tensor<tnsr::ii<DataVector, 3>>(strain_components, num_points);
  const auto stress =
      symmetric_tensor<tnsr::II<DataVector, 3>>(stress_components, num_points);
  tnsr::II<DataVector, 3> stress_from_relation{num_points};
  constitutive_relation.stress(make_not_null(&stress_from_relation), strain, x);
  CHECK_ITERABLE_APPROX(stress_from_relation, stress);
  Scalar<DataVector> bulk_energy{num_points};
  Scalar<DataVector> shear_energy{num_points};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  CHECK_ITERABLE_APPROX(get(bulk_energy),
                        DataVector(num_points, expected_bulk));
  CHECK_ITERABLE_APPROX(get(shear_energy),
                        DataVector(num_points, expected_shear));
  // Total from upstream's -1/2 S_ij T^ij, with the stress recomputed by the
  // constitutive relation
  const auto total_energy =
      Elasticity::potential_energy_density<3>(strain, x, constitutive_relation);
  CHECK_ITERABLE_APPROX(get(total_energy),
                        DataVector(num_points, expected_bulk + expected_shear));
}

void test_pure_dilatation_and_pure_shear() {
  {
    INFO("Pure dilatation");
    // S = eps delta with eps = 1/2: Theta = 3/2, e = 0. T = -K Theta delta =
    // -9/2 delta, Sigma = -27/2. U_bulk = -Theta Sigma / 6 = 27/8
    // = K Theta^2 / 2, U_shear = 0 (e = s = 0 exactly), U_total = 27/8.
    // The densities are O(1), so the default absolute floor of 2.2e-14 bounds
    // any leak of the dilatation into the shear part
    check_isotropic_split({{{0.5, 0., 0.}, {0., 0.5, 0.}, {0., 0., 0.5}}},
                          {{{-4.5, 0., 0.}, {0., -4.5, 0.}, {0., 0., -4.5}}},
                          3.375, 0.);
  }
  {
    INFO("Pure shear");
    // Traceless S = [[1/2, 1/4, -3/8], [1/4, -1/4, 1/8], [-3/8, 1/8, -1/4]],
    // so e = S. T = -2 mu S = [[-4, -2, 3], [-2, 2, -1], [3, -1, 2]],
    // Sigma = 0. e_ij e_ij = 1/4 + 1/16 + 1/16 + 2 (1/16 + 9/64 + 1/64)
    // = 13/16, U_shear = mu e_ij e_ij = 13/4, U_bulk = 0, U_total = 13/4
    check_isotropic_split(
        {{{0.5, 0.25, -0.375}, {0.25, -0.25, 0.125}, {-0.375, 0.125, -0.25}}},
        {{{-4., -2., 3.}, {-2., 2., -1.}, {3., -1., 2.}}}, 0., 3.25);
  }
}

// Near-pure dilatation, isotropic K = 3, mu = 4: S = alpha delta + beta e with
// alpha = 1e4, beta = 2^-11 and e the traceless strain of the pure-shear case
// above (e_ij e_ij = 13/16). Then Theta = 3 alpha, the strain deviator is
// beta e, T = -K Theta delta - 2 mu beta e, and exactly
//   U_bulk = K Theta^2 / 2 = 9 K alpha^2 / 2 = 1.35e9,
//   U_shear = mu beta^2 e_ij e_ij = 2^2 2^-22 13/16 = 13 * 2^-24,
// so the shear density is ~6e-16 of the bulk. U_bulk lies in [2^30, 2^31), so
// its ulp is 2^-22 and every double near it is a multiple of 2^-22. U_shear =
// 3.25 * 2^-22 is off that grid, so a shear density computed as total minus
// bulk errs by at least 0.25 * 2^-22 = U_shear / 13, whatever the summation
// order or FMA contraction. Contracting the deviators loses far less: the
// strain and stress entries (~1e4, ~9e4) carry at most half an ulp (9e-13,
// 7e-12) against deviatoric parts of at least 1.2e-4 and 9.8e-4, i.e. ~1e-8
// relative. The relative tolerance 1e-6 on U_shear therefore rejects any
// route that forms the shear by cancelling the total against the bulk. It
// does not identify the implementation: contracting one deviator against the
// other, raw tensor is algebraically valid and also exact here. The
// comparison uses scale 0
// because Catch's margin is eps * (scale + |expected|), so with scale 1 the
// margin would be ~1e-6, larger than U_shear itself.
void test_bulk_shear_near_pure_dilatation() {
  INFO("Near-pure dilatation: shear density ~6e-16 of the bulk");
  const double alpha = 1.e4;
  const double beta = std::ldexp(1., -11);
  const std::array<std::array<double, 3>, 3> pure_shear{
      {{0.5, 0.25, -0.375}, {0.25, -0.25, 0.125}, {-0.375, 0.125, -0.25}}};
  std::array<std::array<double, 3>, 3> strain_components{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(gsl::at(strain_components, i), j) =
          (i == j ? alpha : 0.) + beta * gsl::at(gsl::at(pure_shear, i), j);
    }
  }
  const size_t num_points = 3;
  const auto strain =
      symmetric_tensor<tnsr::ii<DataVector, 3>>(strain_components, num_points);
  const auto x =
      make_with_value<tnsr::I<DataVector, 3>>(DataVector{num_points}, 0.);
  const Elasticity::ConstitutiveRelations::IsotropicHomogeneous<3>
      constitutive_relation{3., 4.};
  tnsr::II<DataVector, 3> stress{num_points};
  constitutive_relation.stress(make_not_null(&stress), strain, x);
  Scalar<DataVector> bulk_energy{num_points};
  Scalar<DataVector> shear_energy{num_points};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  const double expected_shear = 13. * std::ldexp(1., -24);
  CHECK_ITERABLE_APPROX(get(bulk_energy), DataVector(num_points, 1.35e9));
  Approx relative_approx = Approx::custom().epsilon(1.e-6).scale(0.);
  CHECK_ITERABLE_CUSTOM_APPROX(get(shear_energy),
                               DataVector(num_points, expected_shear),
                               relative_approx);
  // This executes the total-minus-bulk mutation, which the ulp-grid argument
  // above predicts to miss U_shear by at least U_shear / 13, so it must fall
  // outside the tolerance. The total and the bulk density both lie in
  // [2^30, 2^31), so their difference is a multiple of 2^-22 whatever the
  // summation order, and the check does not depend on the order either.
  const auto total_energy =
      Elasticity::potential_energy_density<3>(strain, x, constitutive_relation);
  const DataVector total_minus_bulk = get(total_energy) - get(bulk_energy);
  CHECK(min(abs(total_minus_bulk - expected_shear)) > 1.e-6 * expected_shear);
}

// Cubic material: upstream CubicCrystal with the AlGaAs moduli of
// SingleCoatingMirror.yaml (GPa). For a cubic crystal
// T^aa = -[(c_11 - c_12) S_aa + c_12 Theta] (no sum over a) and
// T^ab = -2 c_44 S_ab for a != b. Hence Sigma = -(c_11 + 2 c_12) Theta and
// s^aa = -(c_11 - c_12) e_aa, so
//   U_bulk = -Theta Sigma / 6 = (c_11 + 2 c_12) Theta^2 / 6,
//   U_shear = (c_11 - c_12) / 2 sum_a e_aa^2 + 2 c_44 sum_{a<b} S_ab^2.
// Unlike the isotropic case, the diagonal and off-diagonal deviatoric parts
// have different moduli (c_11 - c_12 = 64.56, 2 c_44 = 118.3). NumPy
// reproduces both closed forms from a fourth-rank cubic stiffness tensor to
// 7e-16 relative (shear) and 2e-13 relative (bulk, cancellation at small
// Theta) on 1e4 random strains.
void test_bulk_shear_cubic() {
  INFO("Cubic material");
  const double c_11 = 119.94;
  const double c_12 = 55.38;
  const double c_44 = 59.15;
  const Elasticity::ConstitutiveRelations::CubicCrystal constitutive_relation{
      c_11, c_12, c_44};
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{100};
  const auto strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  tnsr::II<DataVector, 3> stress{used_for_size.size()};
  constitutive_relation.stress(make_not_null(&stress), strain, x);
  Scalar<DataVector> bulk_energy{used_for_size.size()};
  Scalar<DataVector> shear_energy{used_for_size.size()};
  Elasticity::bulk_potential_energy_density(make_not_null(&bulk_energy), strain,
                                            stress);
  Elasticity::shear_potential_energy_density(make_not_null(&shear_energy),
                                             strain, stress);
  const DataVector trace_strain =
      get<0, 0>(strain) + get<1, 1>(strain) + get<2, 2>(strain);
  DataVector diagonal_square{used_for_size.size(), 0.};
  for (size_t a = 0; a < 3; ++a) {
    diagonal_square += square(strain.get(a, a) - trace_strain / 3.);
  }
  const DataVector off_diagonal_square = square(get<0, 1>(strain)) +
                                         square(get<0, 2>(strain)) +
                                         square(get<1, 2>(strain));
  const DataVector expected_bulk =
      (c_11 + 2. * c_12) / 6. * square(trace_strain);
  const DataVector expected_shear =
      0.5 * (c_11 - c_12) * diagonal_square + 2. * c_44 * off_diagonal_square;
  // Densities up to ~1e2 GPa for strains in [-1, 1); the classes sum terms in
  // different orders, so the default 2.2e-14 could be hit
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(get(bulk_energy), expected_bulk, custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(shear_energy), expected_shear,
                               custom_approx);
  // Bulk + shear = total, with the total from upstream's -1/2 S_ij T^ij. The
  // shear observer never forms S_ij T^ij (it contracts the deviators), so the
  // two sides are computed by different routes
  const auto total_energy =
      Elasticity::potential_energy_density<3>(strain, x, constitutive_relation);
  const DataVector bulk_plus_shear = get(bulk_energy) + get(shear_energy);
  CHECK_ITERABLE_CUSTOM_APPROX(bulk_plus_shear, get(total_energy),
                               custom_approx);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Elasticity.PotentialEnergy",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions"};
  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_elastic_potential_energy, (2, 3));
  CHECK_FOR_DATAVECTORS(test_compute_tags, (2, 3));
  CHECK_FOR_DATAVECTORS(test_bulk_shear_split, (2, 3));
  CHECK_FOR_DATAVECTORS(test_bulk_shear_random_values, (2, 3));
  test_bulk_shear_fixed_values<2>();
  test_bulk_shear_fixed_values<3>();
  test_bulk_shear_compute_tag_values<2>();
  test_bulk_shear_compute_tag_values<3>();
  test_pure_dilatation_and_pure_shear();
  test_bulk_shear_near_pure_dilatation();
  test_bulk_shear_cubic();
}
