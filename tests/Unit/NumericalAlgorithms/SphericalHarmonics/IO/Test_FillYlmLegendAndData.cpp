// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <typename Generator>
void test(const gsl::not_null<Generator*> generator) {
  std::vector<std::string> legend{};
  std::vector<double> data{};
  const size_t strahlkorper_l_max = 6;
  const size_t larger_l_max = strahlkorper_l_max + 2;
  const double time = 6.7;

  std::uniform_real_distribution<double> dist{0.1, 2.0};

  const size_t radii_size =
      ylm::Spherepack::physical_size(strahlkorper_l_max, strahlkorper_l_max);
  auto radii = make_with_random_values<DataVector>(
      generator, make_not_null(&dist),
      ModalVector{radii_size, std::numeric_limits<double>::signaling_NaN()});

  ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      strahlkorper_l_max, strahlkorper_l_max, radii, std::array{0.9, 0.8, 0.7}};
  ylm::SpherepackIterator iter{strahlkorper_l_max, strahlkorper_l_max};
  std::vector<double> expected_coefs{};
  for (size_t l = 0; l <= strahlkorper_l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      expected_coefs.emplace_back(
          strahlkorper.coefficients()[iter.set(l, m)()]);
    }
  }

  // Where max_l is the same as strahlkorper.l_max()
  {
    ylm::fill_ylm_legend_and_data(make_not_null(&legend), make_not_null(&data),
                                  strahlkorper, time, strahlkorper_l_max);

    // +5 for time, l_max, and 3 components of the center
    const size_t expected_size = expected_coefs.size() + 5;
    CHECK(legend.size() == expected_size);
    CHECK(data.size() == expected_size);
    CHECK(data[0] == time);
    for (size_t i = 0; i < 3; i++) {
      CHECK(data[i + 1] == gsl::at(strahlkorper.expansion_center(), i));
    }
    CHECK(data[4] == strahlkorper_l_max);
    for (size_t i = 5; i < expected_size; i++) {
      CHECK(data[i] == expected_coefs[i - 5]);
    }
  }

  legend.clear();
  data.clear();
  iter.reset();

  // Where max_l is larger than strahlkorper.l_max()
  {
    std::vector<std::string> compare_legend{};
    std::vector<double> compare_data{};
    ylm::fill_ylm_legend_and_data(make_not_null(&legend), make_not_null(&data),
                                  strahlkorper, time, larger_l_max);
    ylm::fill_ylm_legend_and_data(make_not_null(&compare_legend),
                                  make_not_null(&compare_data),
                                  ylm::Strahlkorper<Frame::Inertial>{
                                      larger_l_max, larger_l_max, strahlkorper},
                                  time, larger_l_max);
    CHECK(compare_legend == legend);
    // This is the only thing that should be different so we just check this
    // specifically first, then overwrite it and check the rest of the vector
    CHECK(data[4] == static_cast<double>(strahlkorper_l_max));
    CHECK(compare_data[4] == static_cast<double>(larger_l_max));
    data[4] = static_cast<double>(larger_l_max);
    CHECK(compare_data == data);
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.NumericalAlgorithms.SphericalHarmonics.IO.FillYlmLegendAndData",
    "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
}
