// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/AnisotropicHomogeneous.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/CubicCrystal.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/Factory.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/IsotropicHomogeneous.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace Elasticity::ConstitutiveRelations {
namespace {

// The anisotropic relation is available in the 3D factory only
static_assert(tmpl::list_contains_v<standard_constitutive_relations<3>,
                                    AnisotropicHomogeneous>);
static_assert(not tmpl::list_contains_v<standard_constitutive_relations<2>,
                                        AnisotropicHomogeneous>);

// A general symmetric positive definite Voigt matrix (GPa), with distinct
// shear moduli C_44, C_55, C_66 = 70, 60, 50 (1-based Voigt indices) and
// nonzero normal-shear and shear-shear couplings, so that a dropped factor 2
// or a misplaced Voigt slot changes the stress. Its eigenvalues are 47.88,
// 58.77, 67.71, 115.97, 137.28 and 292.39. Must match `C_GENERAL` in
// AnisotropicHomogeneous.py.
std::array<std::array<double, 6>, 6> c_general() {
  return {{{200., 60., 50., 10., -5., 8.},
           {60., 180., 55., -7., 6., -4.},
           {50., 55., 160., 5., 9., 3.},
           {10., -7., 5., 70., -3., 2.},
           {-5., 6., 9., -3., 60., -4.},
           {8., -4., 3., 2., -4., 50.}}};
}

// Voigt stiffness matrix of a cubic crystal
std::array<std::array<double, 6>, 6> cubic_voigt_matrix(const double c_11,
                                                        const double c_12,
                                                        const double c_44) {
  return {{{c_11, c_12, c_12, 0., 0., 0.},
           {c_12, c_11, c_12, 0., 0., 0.},
           {c_12, c_12, c_11, 0., 0., 0.},
           {0., 0., 0., c_44, 0., 0.},
           {0., 0., 0., 0., c_44, 0.},
           {0., 0., 0., 0., 0., c_44}}};
}

// scale * diag(1, 1, 1, 1, 1, smallest): the eigenvalues of a diagonal
// matrix are its entries, so the eigenvalue ratio is exactly `smallest`.
std::array<std::array<double, 6>, 6> diagonal_voigt_matrix(
    const double scale, const double smallest) {
  std::array<std::array<double, 6>, 6> result{};
  for (size_t i = 0; i < 5; ++i) {
    gsl::at(gsl::at(result, i), i) = scale;
  }
  result[5][5] = scale * smallest;
  return result;
}

// The `StiffnessMatrix` option in YAML, with the key indented by `indent`
// spaces and one flow sequence per row. Entries are printed with 17
// significant digits so that they round trip exactly; non-finite entries use
// the YAML spellings `.nan`, `.inf` and `-.inf`.
std::string stiffness_matrix_yaml(
    const std::array<std::array<double, 6>, 6>& matrix, const size_t indent) {
  std::ostringstream yaml;
  yaml << std::setprecision(17);
  yaml << std::string(indent, ' ') << "StiffnessMatrix:\n";
  for (const auto& row : matrix) {
    yaml << std::string(indent + 2, ' ') << "- [";
    for (size_t j = 0; j < 6; ++j) {
      const double entry = gsl::at(row, j);
      if (std::isnan(entry)) {
        yaml << ".nan";
      } else if (std::isinf(entry)) {
        yaml << (entry > 0. ? ".inf" : "-.inf");
      } else {
        yaml << entry;
      }
      yaml << (j < 5 ? ", " : "]\n");
    }
  }
  return yaml.str();
}

// Free-function wrapper for pypp. The member-function overload of
// `pypp::check_with_random_values` converts the member args to tensors,
// which is implemented only for `double` and `std::array<double, N>`, so the
// 6x6 matrix cannot be forwarded and is hard-coded on both sides instead.
void stress_general(const gsl::not_null<tnsr::II<DataVector, 3>*> stress,
                    const tnsr::ii<DataVector, 3>& strain,
                    const tnsr::I<DataVector, 3>& x) {
  AnisotropicHomogeneous{c_general()}.stress(stress, strain, x);
}

void test_semantics() {
  INFO("Semantics");
  const AnisotropicHomogeneous relation{cubic_voigt_matrix(3., 2., 1.)};
  CHECK(relation == AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., 1.)});
  CHECK(relation != AnisotropicHomogeneous{cubic_voigt_matrix(3., 1., 1.)});
  test_serialization(relation);
  test_copy_semantics(relation);
  const auto created_ptr =
      TestHelpers::test_factory_creation<ConstitutiveRelation<3>,
                                         AnisotropicHomogeneous>(
          "AnisotropicHomogeneous:\n"
          "  StiffnessMatrix:\n"
          "    - [3., 2., 2., 0., 0., 0.]\n"
          "    - [2., 3., 2., 0., 0., 0.]\n"
          "    - [2., 2., 3., 0., 0., 0.]\n"
          "    - [0., 0., 0., 1., 0., 0.]\n"
          "    - [0., 0., 0., 0., 1., 0.]\n"
          "    - [0., 0., 0., 0., 0., 1.]\n")
          ->get_clone();
  REQUIRE(dynamic_cast<const AnisotropicHomogeneous*>(created_ptr.get()) !=
          nullptr);
  CHECK(dynamic_cast<const AnisotropicHomogeneous&>(*created_ptr) == relation);
  AnisotropicHomogeneous moved_relation{cubic_voigt_matrix(3., 2., 1.)};
  test_move_semantics(std::move(moved_relation), relation);

  // The cubic matrix above has zeros in all coupling entries, so repeat the
  // round trips with the general matrix to exercise them
  const AnisotropicHomogeneous general_relation{c_general()};
  test_serialization(general_relation);
  const auto created_general_ptr =
      TestHelpers::test_factory_creation<ConstitutiveRelation<3>,
                                         AnisotropicHomogeneous>(
          "AnisotropicHomogeneous:\n"
          "  StiffnessMatrix:\n"
          "    - [200., 60., 50., 10., -5., 8.]\n"
          "    - [60., 180., 55., -7., 6., -4.]\n"
          "    - [50., 55., 160., 5., 9., 3.]\n"
          "    - [10., -7., 5., 70., -3., 2.]\n"
          "    - [-5., 6., 9., -3., 60., -4.]\n"
          "    - [8., -4., 3., 2., -4., 50.]\n")
          ->get_clone();
  REQUIRE(dynamic_cast<const AnisotropicHomogeneous*>(
              created_general_ptr.get()) != nullptr);
  const auto& created_general =
      dynamic_cast<const AnisotropicHomogeneous&>(*created_general_ptr);
  CHECK(created_general == general_relation);
  CHECK(created_general.stiffness_matrix() == c_general());
  // Equality sees a change in a single coupling pair
  auto changed_coupling = c_general();
  changed_coupling[0][5] = changed_coupling[5][0] = 7.;
  CHECK(general_relation != AnisotropicHomogeneous{changed_coupling});
}

void test_consistency_with_cubic_crystal() {
  INFO("Consistency with CubicCrystal");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{10};
  const auto strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  // Silicon: c_11=166, c_12=64, c_44=80 (see Test_CubicCrystal.cpp)
  const AnisotropicHomogeneous anisotropic{cubic_voigt_matrix(166., 64., 80.)};
  const CubicCrystal cubic{166., 64., 80.};
  tnsr::II<DataVector, 3> anisotropic_stress{used_for_size.size()};
  tnsr::II<DataVector, 3> cubic_stress{used_for_size.size()};
  anisotropic.stress(make_not_null(&anisotropic_stress), strain, x);
  cubic.stress(make_not_null(&cubic_stress), strain, x);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(anisotropic_stress.get(i, j),
                            cubic_stress.get(i, j));
    }
  }
}

// An isotropic Voigt matrix built from the bulk and shear moduli,
// C_11 = K + 4 mu / 3, C_12 = K - 2 mu / 3, C_44 = mu
void test_consistency_with_isotropic_homogeneous(const double bulk_modulus,
                                                 const double shear_modulus) {
  CAPTURE(bulk_modulus);
  CAPTURE(shear_modulus);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{10};
  const auto strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const double c_11 = bulk_modulus + 4. / 3. * shear_modulus;
  const double c_12 = bulk_modulus - 2. / 3. * shear_modulus;
  const AnisotropicHomogeneous anisotropic{
      cubic_voigt_matrix(c_11, c_12, shear_modulus)};
  const IsotropicHomogeneous<3> isotropic{bulk_modulus, shear_modulus};
  tnsr::II<DataVector, 3> anisotropic_stress{used_for_size.size()};
  tnsr::II<DataVector, 3> isotropic_stress{used_for_size.size()};
  anisotropic.stress(make_not_null(&anisotropic_stress), strain, x);
  isotropic.stress(make_not_null(&isotropic_stress), strain, x);
  // The two classes sum terms of size ~C S ~ 1e2 in different orders, so a
  // component near zero can differ by a few 1e-14 in absolute terms
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK_ITERABLE_CUSTOM_APPROX(anisotropic_stress.get(i, j),
                                   isotropic_stress.get(i, j), custom_approx);
    }
  }
}

void test_consistency_with_isotropic_homogeneous() {
  INFO("Consistency with IsotropicHomogeneous");
  // Fused silica (GPa): E = 72.7, nu = 0.167
  test_consistency_with_isotropic_homogeneous(36.38638638638639,
                                              31.148243359040272);
  // Iron: E=100, nu=0.29 (see Test_IsotropicHomogeneous.cpp)
  test_consistency_with_isotropic_homogeneous(79.3651, 38.7597);
}

// Pure shear strain S_xy = S_yx = s probes only the last column of the Voigt
// matrix, through the engineering strain 2 s: T^ij = -2 s C_{V(ij),6}, with
// V(ij) the 1-based Voigt index of ij
void test_engineering_shear() {
  INFO("Engineering shear and coupling slots");
  const DataVector used_for_size{1};
  auto strain = make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, 0.);
  get<0, 1>(strain) = 1.e-3;
  const auto x = make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.);
  tnsr::II<DataVector, 3> stress{used_for_size.size()};
  AnisotropicHomogeneous{c_general()}.stress(make_not_null(&stress), strain, x);
  // Reference values (GPa) from the contraction -C_ijkl S_kl of the full
  // fourth-rank tensor (`stiffness_tensor` in AnisotropicHomogeneous.py),
  // also exact in rational arithmetic
  CHECK_ITERABLE_APPROX(stress.get(0, 1)[0], -0.1);    // -2 C_66 s
  CHECK_ITERABLE_APPROX(stress.get(0, 0)[0], -0.016);  // -2 C_16 s
  CHECK_ITERABLE_APPROX(stress.get(1, 1)[0], 0.008);   // -2 C_26 s
  CHECK_ITERABLE_APPROX(stress.get(2, 2)[0], -0.006);  // -2 C_36 s
  CHECK_ITERABLE_APPROX(stress.get(1, 2)[0], -0.004);  // -2 C_46 s
  CHECK_ITERABLE_APPROX(stress.get(0, 2)[0], 0.008);   // -2 C_56 s
}

void test_fixed_strain() {
  INFO("Fixed-strain regression");
  const DataVector used_for_size{1};
  // S = [[1, 0.3, -0.2], [0.3, -0.5, 0.4], [-0.2, 0.4, 0.7]] * 1e-3
  auto strain = make_with_value<tnsr::ii<DataVector, 3>>(used_for_size, 0.);
  get<0, 0>(strain) = 1.e-3;
  get<0, 1>(strain) = 0.3e-3;
  get<0, 2>(strain) = -0.2e-3;
  get<1, 1>(strain) = -0.5e-3;
  get<1, 2>(strain) = 0.4e-3;
  get<2, 2>(strain) = 0.7e-3;
  const auto x = make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.);
  tnsr::II<DataVector, 3> stress{used_for_size.size()};
  AnisotropicHomogeneous{c_general()}.stress(make_not_null(&stress), strain, x);
  // Reference T^ij = -C_ijkl S_kl (GPa) from the full fourth-rank tensor.
  // The values are exact: rational arithmetic gives -1099/5000, -453/10000,
  // 61/2000, 19/10000, -377/5000 and -1367/10000, and NumPy reproduces them
  // to 3e-17.
  const std::array<std::array<double, 3>, 3> expected_stress{
      {{-0.2198, -0.0453, 0.0305},
       {-0.0453, 0.0019, -0.0754},
       {0.0305, -0.0754, -0.1367}}};
  double energy_density = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CAPTURE(i);
      CAPTURE(j);
      CHECK_ITERABLE_APPROX(stress.get(i, j)[0],
                            gsl::at(gsl::at(expected_stress, i), j));
      energy_density -= 0.5 * stress.get(i, j)[0] * strain.get(i, j)[0];
    }
  }
  // Energy density -T^ij S_ij / 2 = 20807/1e8 GPa = 2.0807e5 J/m^3 (exact),
  // compared against the fixed value
  CHECK_ITERABLE_APPROX(energy_density, 2.0807e-4);
}

void test_general_matrix() {
  INFO("General matrix: comparison to an independent Python contraction");
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Elasticity/ConstitutiveRelations");
  pypp::check_with_random_values<1>(&stress_general, "AnisotropicHomogeneous",
                                    {"stress_general"}, {{{-1., 1.}}},
                                    DataVector(5));
}

void test_energy_positivity() {
  INFO("Positive strain energy");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);
  const DataVector used_for_size{100};
  const auto strain = make_with_random_values<tnsr::ii<DataVector, 3>>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto x = make_with_value<tnsr::I<DataVector, 3>>(used_for_size, 0.);
  tnsr::II<DataVector, 3> stress{used_for_size.size()};
  AnisotropicHomogeneous{c_general()}.stress(make_not_null(&stress), strain, x);
  auto energy_density = make_with_value<DataVector>(used_for_size, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      energy_density -= 0.5 * stress.get(i, j) * strain.get(i, j);
    }
  }
  CHECK(min(energy_density) > 0.);
}

void test_errors() {
  INFO("Invalid stiffness matrices");
  auto asymmetric = cubic_voigt_matrix(3., 2., 1.);
  asymmetric[0][1] = 2.5;
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{asymmetric},
      Catch::Matchers::ContainsSubstring("must be symmetric in Voigt"));
  // The criterion is that all eigenvalues of the Voigt matrix are positive.
  // For a cubic crystal they are c_11 + 2 c_12 (hydrostatic, once),
  // c_11 - c_12 (twice) and c_44 (three times), and each case below makes
  // exactly one of them non-positive.
  // c_11 - c_12 = -1 < 0. A negative eigenvalue is reported as physical
  // instability, not as ill-conditioning (see test_threshold).
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(2., 3., 1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite") and
          Catch::Matchers::ContainsSubstring("would be unstable"));
  // Hydrostatic instability: c_11 + 2 c_12 = -1 < 0, although
  // c_11 > |c_12|, so that condition alone is not sufficient
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(2., -1.5, 1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
  // c_44 = -1 < 0
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., -1.)},
      Catch::Matchers::ContainsSubstring("must be positive definite"));
  // Positive semidefinite: c_44 = 0 gives a zero eigenvalue, which is
  // non-positive and therefore reported as unstable (independent of the
  // relative threshold). Only the message is checked, not the value.
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{cubic_voigt_matrix(3., 2., 0.)},
      Catch::Matchers::ContainsSubstring("must be positive definite") and
          Catch::Matchers::ContainsSubstring("would be unstable"));
}

// The positive-definiteness check is relative: the smallest eigenvalue must
// exceed `min_relative_eigenvalue` times the largest. Diagonal fixtures
// make the eigenvalues exact, so the boundary can be tested on both sides.
void test_threshold() {
  INFO("Relative positive-definiteness threshold");
  using Catch::Matchers::ContainsSubstring;
  static_assert(AnisotropicHomogeneous::min_relative_eigenvalue == 1.e-12);
  // The same outcomes after uniform rescaling by 1e9 (Pa instead of GPa)
  // and by 1e-9 show that the threshold is relative, not absolute.
  for (const double scale : {1., 1.e9, 1.e-9}) {
    CAPTURE(scale);
    // Ratio below the threshold: positive definite, but rejected as too
    // ill-conditioned (not as unstable)
    CHECK_THROWS_WITH(
        AnisotropicHomogeneous{diagonal_voigt_matrix(scale, 0.5e-12)},
        ContainsSubstring("too ill-conditioned") and
            not ContainsSubstring("would be unstable"));
    // Exactly at the threshold: the inequality is strict, so this is
    // rejected. Both sides of the comparison are the product of the same
    // two doubles (1e-12 and the scale), so there is no roundoff margin
    // as long as LAPACK returns the diagonal entries exactly.
    CHECK_THROWS_WITH(
        AnisotropicHomogeneous{diagonal_voigt_matrix(scale, 1.e-12)},
        ContainsSubstring("too ill-conditioned"));
    // Ratio above the threshold: accepted
    CHECK_NOTHROW(AnisotropicHomogeneous(diagonal_voigt_matrix(scale, 2.e-12)));
  }
  // Non-finite entries are rejected before the eigenvalue computation,
  // with a message that names the entry
  auto with_nan = cubic_voigt_matrix(3., 2., 1.);
  with_nan[0][0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{with_nan},
      ContainsSubstring("must be finite") and ContainsSubstring("C[0][0]=nan"));
  auto with_inf = cubic_voigt_matrix(3., 2., 1.);
  with_inf[5][5] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_WITH(
      AnisotropicHomogeneous{with_inf},
      ContainsSubstring("must be finite") and ContainsSubstring("C[5][5]=inf"));
}

// The constructor checks also reject invalid matrices given in an input
// file, with the parser's context ("In string", "While creating a ...")
// prepended to the message. Only stable fragments are matched, no line or
// column numbers.
void test_option_parsing_errors() {
  INFO("Rejection through option parsing");
  using Catch::Matchers::ContainsSubstring;
  auto with_nan = c_general();
  with_nan[2][2] = std::numeric_limits<double>::quiet_NaN();
  auto asymmetric = c_general();
  asymmetric[0][5] = 9.;
  const std::array<std::pair<std::array<std::array<double, 6>, 6>, std::string>,
                   4>
      cases{{{with_nan, "must be finite"},
             {asymmetric, "must be symmetric"},
             {cubic_voigt_matrix(2., 3., 1.), "must be positive definite"},
             {diagonal_voigt_matrix(1., 0.5e-12), "too ill-conditioned"}}};
  for (const auto& [matrix, fragment] : cases) {
    CAPTURE(fragment);
    CHECK_THROWS_WITH(TestHelpers::test_creation<AnisotropicHomogeneous>(
                          stiffness_matrix_yaml(matrix, 0)),
                      ContainsSubstring(fragment) and
                          ContainsSubstring("StiffnessMatrix") and
                          ContainsSubstring("In string") and
                          ContainsSubstring("While creating"));
    CHECK_THROWS_WITH(
        (TestHelpers::test_factory_creation<ConstitutiveRelation<3>,
                                            AnisotropicHomogeneous>(
            "AnisotropicHomogeneous:\n" + stiffness_matrix_yaml(matrix, 2))),
        ContainsSubstring(fragment) and ContainsSubstring("StiffnessMatrix") and
            ContainsSubstring("While operating factory for") and
            ContainsSubstring("While creating"));
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Elasticity.ConstitutiveRelations.AnisotropicHomogeneous",
    "[PointwiseFunctions][Unit][Elasticity]") {
  test_semantics();
  test_consistency_with_cubic_crystal();
  test_consistency_with_isotropic_homogeneous();
  test_engineering_shear();
  test_fixed_strain();
  test_general_matrix();
  test_energy_positivity();
  test_errors();
  test_threshold();
  test_option_parsing_errors();
}

}  // namespace Elasticity::ConstitutiveRelations
