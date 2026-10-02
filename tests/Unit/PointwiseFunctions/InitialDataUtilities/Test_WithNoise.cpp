// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/WithNoise.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {

using evolution::initial_data::NoiseAmplitudeType;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::PowX<1, Frame::Inertial>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<ScalarWave::Solutions::PlaneWave<1>,
                              evolution::initial_data::WithNoise>>>;
  };
};

void test_add_noise_to_tensor() {
  const size_t n_pts = 5;
  {
    INFO("Amplitude 0 is a no-op");
    Scalar<DataVector> s{DataVector(n_pts, 3.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s), 0.0, NoiseAmplitudeType::Absolute, 6, 0);
    CHECK(get(s) == DataVector(n_pts, 3.0));
  }
  {
    INFO("Non-zero amplitude changes value");
    Scalar<DataVector> s{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s), 1.0, NoiseAmplitudeType::Absolute, 7, 0);
    bool any_nonzero = false;
    for (const double v : get(s)) {
      CHECK(std::abs(v) <= 1.0);
      if (v != 0.0) {
        any_nonzero = true;
      }
    }
    CHECK(any_nonzero);
  }
  {
    INFO("Same seed and offset -> identical noise (reproducibility)");
    Scalar<DataVector> s1{DataVector(n_pts, 0.0)};
    Scalar<DataVector> s2{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s1), 1.0, NoiseAmplitudeType::Absolute, 99, 0);
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s2), 1.0, NoiseAmplitudeType::Absolute, 99, 0);
    CHECK(get(s1) == get(s2));
  }
  {
    INFO("Different seeds -> different noise");
    Scalar<DataVector> s1{DataVector(n_pts, 0.0)};
    Scalar<DataVector> s2{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s1), 1.0, NoiseAmplitudeType::Absolute, 10, 0);
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s2), 1.0, NoiseAmplitudeType::Absolute, 11, 0);
    CHECK(get(s1) != get(s2));
  }
  {
    INFO(
        "Different component offsets -> different noise on separate tensor "
        "fields");
    tnsr::i<DataVector, 2> v1{DataVector(n_pts, 0.0)};
    tnsr::i<DataVector, 2> v2{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&v1), 1.0, NoiseAmplitudeType::Absolute, 42, 0);
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&v2), 1.0, NoiseAmplitudeType::Absolute, 42, 2);
    CHECK(get<0>(v1) != get<0>(v2));
    CHECK(get<1>(v1) != get<1>(v2));
  }
  {
    INFO(
        "Noise values are within amplitude bounds and tensor-type independent");
    const double amplitude = 2.5;
    tnsr::ii<DataVector, 2> sym_tensor{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&sym_tensor), amplitude, NoiseAmplitudeType::Absolute, 13,
        0);
    for (size_t a = 0; a < 2; ++a) {
      for (size_t b = a; b < 2; ++b) {
        for (const double v : sym_tensor.get(a, b)) {
          CHECK(std::abs(v) <= amplitude);
        }
      }
    }
  }
}

void test_relative_noise() {
  const size_t n_pts = 20;
  const double amplitude = 0.1;
  {
    INFO("Relative noise is bounded by amplitude times the tensor max");
    // Components of very different magnitude, including an identically zero
    // one, which must still be perturbed.
    tnsr::i<DataVector, 3> v{n_pts};
    for (size_t i = 0; i < n_pts; ++i) {
      get<0>(v)[i] = 1.0e3 * static_cast<double>(i);
      get<1>(v)[i] = -2.0e3 + static_cast<double>(i);
    }
    get<2>(v) = 0.0;
    const double max_abs = 1.0e3 * static_cast<double>(n_pts - 1);
    const auto clean = v;
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&v), amplitude, NoiseAmplitudeType::Relative, 5, 0);
    for (size_t d = 0; d < 3; ++d) {
      CHECK(v.get(d) != clean.get(d));
      CHECK(max(abs(v.get(d) - clean.get(d))) <= amplitude * max_abs);
    }
    // With so many draws, the noise must exceed the absolute amplitude bound.
    CHECK(max(abs(get<2>(v))) > amplitude);
  }
  {
    INFO("Relative noise scales linearly with the tensor");
    Scalar<DataVector> s1{DataVector(n_pts, 0.0)};
    for (size_t i = 0; i < n_pts; ++i) {
      get(s1)[i] = std::sin(static_cast<double>(i));
    }
    const double scale = 1.0e-5;
    Scalar<DataVector> s2{DataVector(scale * get(s1))};
    const auto clean1 = s1;
    const auto clean2 = s2;
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s1), amplitude, NoiseAmplitudeType::Relative, 8, 0);
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s2), amplitude, NoiseAmplitudeType::Relative, 8, 0);
    CHECK_ITERABLE_APPROX(DataVector(scale * (get(s1) - get(clean1))),
                          DataVector(get(s2) - get(clean2)));
  }
  {
    INFO(
        "Absolute noise is independent of the tensor, relative noise is the "
        "absolute noise times the tensor max");
    DataVector values(n_pts);
    for (size_t i = 0; i < n_pts; ++i) {
      values[i] = 3.0 * std::sin(static_cast<double>(i));
    }
    const double scale = 1.0e-5;
    const double max_abs = max(abs(values));
    const auto noise = [&](const DataVector& data,
                           const NoiseAmplitudeType amplitude_type) {
      Scalar<DataVector> s{data};
      evolution::initial_data::add_noise_to_tensor(make_not_null(&s), amplitude,
                                                   amplitude_type, 12, 0);
      return DataVector(get(s) - data);
    };
    const DataVector absolute_noise =
        noise(values, NoiseAmplitudeType::Absolute);
    // Same seed, tensor rescaled: absolute noise is unchanged.
    CHECK_ITERABLE_APPROX(absolute_noise, noise(DataVector(scale * values),
                                                NoiseAmplitudeType::Absolute));
    // Same seed and draws, interval rescaled by the tensor max.
    CHECK_ITERABLE_APPROX(DataVector(max_abs * absolute_noise),
                          noise(values, NoiseAmplitudeType::Relative));
  }
  {
    INFO("Relative noise on an identically zero tensor is a no-op");
    Scalar<DataVector> s{DataVector(n_pts, 0.0)};
    evolution::initial_data::add_noise_to_tensor(
        make_not_null(&s), amplitude, NoiseAmplitudeType::Relative, 9, 0);
    CHECK(get(s) == DataVector(n_pts, 0.0));
  }
  {
    INFO("Streaming NoiseAmplitudeType");
    CHECK(get_output(NoiseAmplitudeType::Absolute) == "Absolute");
    CHECK(get_output(NoiseAmplitudeType::Relative) == "Relative");
  }
}

void test_make_element_seed() {
  const size_t n_pts = 3;
  tnsr::I<DataVector, 2, Frame::Inertial> coords_a{n_pts};
  tnsr::I<DataVector, 2, Frame::Inertial> coords_b{n_pts};
  for (size_t i = 0; i < n_pts; ++i) {
    coords_a.get(0)[i] = 1.0 + static_cast<double>(i);
    coords_a.get(1)[i] = 2.0 + static_cast<double>(i);
    coords_b.get(0)[i] =
        9.0 + static_cast<double>(i);  // different first grid point
    coords_b.get(1)[i] = 2.0 + static_cast<double>(i);
  }
  {
    INFO("Same base seed and coords -> same element seed (reproducibility)");
    CHECK(evolution::initial_data::make_element_seed(42, coords_a) ==
          evolution::initial_data::make_element_seed(42, coords_a));
  }
  {
    INFO("Different base seeds -> different element seeds");
    CHECK(evolution::initial_data::make_element_seed(42, coords_a) !=
          evolution::initial_data::make_element_seed(43, coords_a));
  }
  {
    INFO("Different first grid point -> different element seed");
    CHECK(evolution::initial_data::make_element_seed(42, coords_a) !=
          evolution::initial_data::make_element_seed(42, coords_b));
  }
}

void test_unwrap() {
  // Single WithNoise: unwrap() reaches the inner solution.
  auto plane_wave = std::make_unique<ScalarWave::Solutions::PlaneWave<1>>(
      std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.0}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  const auto* const plane_wave_ptr = plane_wave.get();
  const evolution::initial_data::WithNoise with_noise{
      std::move(plane_wave), /*amplitude=*/1.0e-4, NoiseAmplitudeType::Absolute,
      /*seed=*/size_t{0},
      /*variables=*/std::vector<std::string>{"All"}};
  CHECK(&with_noise.unwrap() == plane_wave_ptr);

  // Nested WithNoise: construction must be rejected at the outer wrapper.
  auto inner = std::make_unique<ScalarWave::Solutions::PlaneWave<1>>(
      std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.0}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(1));
  CHECK_THROWS_WITH(
      (evolution::initial_data::WithNoise{
          std::make_unique<evolution::initial_data::WithNoise>(
              std::move(inner), 1.0e-4, NoiseAmplitudeType::Absolute, 0_st,
              std::vector<std::string>{"All"}),
          1.0e-4, NoiseAmplitudeType::Absolute, 0_st,
          std::vector<std::string>{"All"}}),
      Catch::Matchers::ContainsSubstring(
          "WithNoise cannot wrap another WithNoise"));
}

void test_with_noise_construction() {
  // Directly construct a WithNoise with a PlaneWave inner solution
  auto plane_wave = std::make_unique<ScalarWave::Solutions::PlaneWave<1>>(
      std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.0}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  const evolution::initial_data::WithNoise with_noise{
      std::move(plane_wave), 1.0e-4, NoiseAmplitudeType::Relative, 43_st,
      std::vector<std::string>{"Psi", "Pi"}};

  CHECK(with_noise.amplitude() == approx(1.0e-4));
  CHECK(with_noise.amplitude_type() == NoiseAmplitudeType::Relative);
  CHECK(with_noise.seed() == 43);
  CHECK(with_noise.variables() == std::vector<std::string>{"Psi", "Pi"});

  test_copy_semantics(with_noise);
}

void test_with_noise_serialization() {
  auto plane_wave = std::make_unique<ScalarWave::Solutions::PlaneWave<1>>(
      std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.0}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(2));
  const evolution::initial_data::WithNoise with_noise{
      std::move(plane_wave), 1.0e-4, NoiseAmplitudeType::Relative, 44_st,
      std::vector<std::string>{"All"}};

  test_serialization(with_noise);

  // Test round-trip via unique_ptr base type
  const auto cloned = with_noise.get_clone();
  const auto& cloned_wn =
      dynamic_cast<const evolution::initial_data::WithNoise&>(*cloned);
  CHECK(cloned_wn.amplitude() == approx(with_noise.amplitude()));
  CHECK(cloned_wn.amplitude_type() == with_noise.amplitude_type());
  CHECK(cloned_wn.seed() == with_noise.seed());
  CHECK(cloned_wn.variables() == with_noise.variables());
}

// Verify that the selective variable mechanism correctly applies noise to only
// the named fields and leaves the others untouched, using the same building
// blocks (add_noise_to_tensor + variables() list) that the actions use.
void test_selective_variable_noise() {
  const size_t n_pts = 4;
  const size_t element_seed = 77;
  const double amplitude = 1.0;

  auto plane_wave = std::make_unique<ScalarWave::Solutions::PlaneWave<1>>(
      std::array<double, 1>{{1.0}}, std::array<double, 1>{{0.0}},
      std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(1));
  const evolution::initial_data::WithNoise wn{
      std::move(plane_wave), amplitude, NoiseAmplitudeType::Absolute, size_t{0},
      std::vector<std::string>{"Psi"}};

  // Simulate action selection: two scalars with tag names "Psi" and "Pi".
  Scalar<DataVector> psi{DataVector(n_pts, 5.0)};
  Scalar<DataVector> pi{DataVector(n_pts, 5.0)};
  const auto& targets = wn.variables();
  size_t offset = 0;
  if (alg::found(targets, std::string{"Psi"})) {
    evolution::initial_data::add_noise_to_tensor(make_not_null(&psi), amplitude,
                                                 NoiseAmplitudeType::Absolute,
                                                 element_seed, offset);
  }
  offset += Scalar<DataVector>::size();
  if (alg::found(targets, std::string{"Pi"})) {
    evolution::initial_data::add_noise_to_tensor(make_not_null(&pi), amplitude,
                                                 NoiseAmplitudeType::Absolute,
                                                 element_seed, offset);
  }

  // "Psi" is in variables() -> must be perturbed
  bool psi_changed = false;
  for (const double v : get(psi)) {
    if (v != 5.0) {
      psi_changed = true;
    }
  }
  CHECK(psi_changed);
  // "Pi" is not in variables() -> must be unchanged
  CHECK(get(pi) == DataVector(n_pts, 5.0));
}

void test_with_noise_option_parsing() {
  {
    const auto created =
        TestHelpers::test_creation<evolution::initial_data::WithNoise,
                                   Metavariables>(
            "Amplitude: 1.0e-6\n"
            "AmplitudeType: Absolute\n"
            "Seed: 41\n"
            "Variables: [Psi, Pi]\n"
            "Solution:\n"
            "  PlaneWave:\n"
            "    WaveVector: [1.0]\n"
            "    Center: [0.0]\n"
            "    Profile:\n"
            "      PowX:\n"
            "        Power: 2\n");
    CHECK(created.amplitude() == approx(1.0e-6));
    CHECK(created.amplitude_type() == NoiseAmplitudeType::Absolute);
    CHECK(created.seed() == 41);
    CHECK(created.variables() == std::vector<std::string>{"Psi", "Pi"});
  }
  {
    INFO("Test with `All`");
    const auto created =
        TestHelpers::test_creation<evolution::initial_data::WithNoise,
                                   Metavariables>(
            "Amplitude: 0.5\n"
            "AmplitudeType: Relative\n"
            "Seed: None\n"
            "Variables: [All]\n"
            "Solution:\n"
            "  PlaneWave:\n"
            "    WaveVector: [1.0]\n"
            "    Center: [0.0]\n"
            "    Profile:\n"
            "      PowX:\n"
            "        Power: 1\n");
    CHECK(created.amplitude() == approx(0.5));
    CHECK(created.amplitude_type() == NoiseAmplitudeType::Relative);
    CHECK(created.variables() == std::vector<std::string>{"All"});
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.InitialDataUtilities.WithNoise",
                  "[Unit][PointwiseFunctions]") {
  register_factory_classes_with_charm<Metavariables>();
  test_add_noise_to_tensor();
  test_relative_noise();
  test_make_element_seed();
  test_unwrap();
  test_with_noise_construction();
  test_with_noise_serialization();
  test_with_noise_option_parsing();
  test_selective_variable_noise();
}
