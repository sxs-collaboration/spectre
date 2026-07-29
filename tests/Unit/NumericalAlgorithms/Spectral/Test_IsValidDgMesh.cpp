// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/IsValidDgMesh.hpp"
#include "NumericalAlgorithms/Spectral/Limits.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace Spectral {
namespace {
constexpr auto i1_bases = std::array{Basis::Legendre, Basis::Chebyshev};
constexpr auto i1_quadratures =
    std::array{Quadrature::Gauss, Quadrature::GaussLobatto};

size_t low_extent(const Basis basis, const Quadrature quadrature) {
  return limits::min(basis, quadrature) - 1;
}

size_t high_extent(const Basis basis, const Quadrature quadrature) {
  return limits::max(basis, quadrature) + 1;
}

size_t random_extent(const gsl::not_null<std::mt19937*> generator,
                     const Basis basis, const Quadrature quadrature) {
  const auto min = limits::min(basis, quadrature);
  const auto max = limits::max(basis, quadrature);
  if (max < min) {
    // This could happen for an invalid basis and quadrature, so return max
    // Otherwise the distribution will be invalid and, either throw, or
    // return a value higher than 255 which triggers an ASSERT in Mesh
    return max;
  }
  std::uniform_int_distribution<size_t> distrib(min, max);
  if (quadrature == Quadrature::Equiangular) {
    const size_t result = distrib(*generator);
    return result % 2 == 0 ? result + 1 : result;
  }
  return distrib(*generator);
}

void test_1d(const gsl::not_null<std::mt19937*> generator) {
  std::vector<std::pair<Basis, Quadrature>> valid_basis_and_quadratures;
  valid_basis_and_quadratures.reserve(5);
  valid_basis_and_quadratures.emplace_back(Basis::Fourier,
                                           Quadrature::Equiangular);
  for (const auto basis : i1_bases) {
    for (const auto quadrature : i1_quadratures) {
      valid_basis_and_quadratures.emplace_back(basis, quadrature);
    }
  }

  for (const auto basis : all_bases()) {
    if (basis == Basis::Uninitialized) {
      continue;
    }
    CAPTURE(basis);
    for (const auto quadrature : all_quadratures()) {
      if (quadrature == Quadrature::Uninitialized) {
        continue;
      }
      CAPTURE(quadrature);
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<1>(low_extent(basis, quadrature), basis, quadrature)));
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<1>(high_extent(basis, quadrature), basis, quadrature)));
      if (alg::found(valid_basis_and_quadratures,
                     std::pair{basis, quadrature})) {
        CHECK(is_valid_dg_mesh(Mesh<1>(
            random_extent(generator, basis, quadrature), basis, quadrature)));
      } else {
        CHECK_FALSE(is_valid_dg_mesh(Mesh<1>(
            random_extent(generator, basis, quadrature), basis, quadrature)));
      }
    }
  }
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<1>{4, Basis::Fourier, Quadrature::Equiangular}));
}

void test_2d(const gsl::not_null<std::mt19937*> generator) {
  std::vector<std::pair<std::array<Basis, 2>, std::array<Quadrature, 2>>>
      valid_basis_and_quadratures;
  valid_basis_and_quadratures.reserve(27);
  valid_basis_and_quadratures.emplace_back(bases::hypertorus<2>,
                                           quadratures::hypertorus<2>);
  valid_basis_and_quadratures.emplace_back(bases::spherical_surface,
                                           quadratures::spherical_surface);
  valid_basis_and_quadratures.emplace_back(bases::disk, quadratures::disk);
  for (const auto i1_basis : i1_bases) {
    for (const auto i1_quadrature : i1_quadratures) {
      for (const auto another_i1_basis : i1_bases) {
        for (const auto another_i1_quadrature : i1_quadratures) {
          valid_basis_and_quadratures.emplace_back(
              std::array{i1_basis, another_i1_basis},
              std::array{i1_quadrature, another_i1_quadrature});
        }
      }
      valid_basis_and_quadratures.emplace_back(
          std::array{i1_basis, Basis::Fourier},
          std::array{i1_quadrature, Quadrature::Equiangular});
      valid_basis_and_quadratures.emplace_back(
          std::array{Basis::Fourier, i1_basis},
          std::array{Quadrature::Equiangular, i1_quadrature});
    }
  }

  CHECK(valid_basis_and_quadratures.size() == 27);

  std::vector<std::array<Basis, 2>> bases;
  bases.reserve(81);
  for (const auto xi_basis : all_bases()) {
    if (xi_basis == Basis::Uninitialized) {
      continue;
    }
    for (const auto eta_basis : all_bases()) {
      if (eta_basis == Basis::Uninitialized) {
        continue;
      }
      bases.emplace_back(std::array{xi_basis, eta_basis});
    }
  }
  CHECK(bases.size() == 81);

  std::vector<std::array<Quadrature, 2>> quadratures;
  quadratures.reserve(81);
  for (const auto xi_quadrature : all_quadratures()) {
    if (xi_quadrature == Quadrature::Uninitialized) {
      continue;
    }
    for (const auto eta_quadrature : all_quadratures()) {
      if (eta_quadrature == Quadrature::Uninitialized) {
        continue;
      }
      quadratures.emplace_back(std::array{xi_quadrature, eta_quadrature});
    }
  }
  CHECK(quadratures.size() == 81);

  for (const auto basis : bases) {
    CAPTURE(basis);
    for (const auto quadrature : quadratures) {
      CAPTURE(quadrature);
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<2>(std::array{low_extent(basis[0], quadrature[0]), 5_st}, basis,
                  quadrature)));
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<2>(std::array{5_st, high_extent(basis[1], quadrature[1])}, basis,
                  quadrature)));
      if (alg::found(valid_basis_and_quadratures,
                     std::pair{basis, quadrature})) {
        if (basis == bases::spherical_surface) {
          const size_t nth = random_extent(generator, basis[0], quadrature[0]);
          CHECK(is_valid_dg_mesh(
              Mesh<2>(std::array{nth, 2 * nth - 1}, basis, quadrature)));
        } else if (basis == bases::disk) {
          const size_t nr = random_extent(generator, basis[0], quadrature[0]);
          CHECK(is_valid_dg_mesh(
              Mesh<2>(std::array{nr, 4 * nr - 3}, basis, quadrature)));
        } else {
          CHECK(is_valid_dg_mesh(Mesh<2>(
              std::array{random_extent(generator, basis[0], quadrature[0]),
                         random_extent(generator, basis[1], quadrature[1])},
              basis, quadrature)));
        }
      } else {
        CHECK_FALSE(is_valid_dg_mesh(Mesh<2>(
            std::array{random_extent(generator, basis[0], quadrature[0]),
                       random_extent(generator, basis[1], quadrature[1])},
            basis, quadrature)));
      }
    }
  }
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{4, Basis::Fourier, Quadrature::Equiangular}));
  CHECK_FALSE(is_valid_dg_mesh(Mesh<2>{
      {4_st, 9_st}, bases::spherical_surface, quadratures::spherical_surface}));
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<2>{{4_st, 9_st}, bases::disk, quadratures::disk}));
}

void test_3d(const gsl::not_null<std::mt19937*> generator) {
  std::vector<std::pair<std::array<Basis, 3>, std::array<Quadrature, 3>>>
      valid_basis_and_quadratures;
  valid_basis_and_quadratures.reserve(159);
  valid_basis_and_quadratures.emplace_back(bases::hypertorus<3>,
                                           quadratures::hypertorus<3>);
  valid_basis_and_quadratures.emplace_back(bases::full_sphere,
                                           quadratures::full_sphere);
  valid_basis_and_quadratures.emplace_back(bases::cartoon_sphere_inner,
                                           quadratures::cartoon_sphere_inner);
  for (const auto i1_basis : i1_bases) {
    for (const auto i1_quadrature : i1_quadratures) {
      for (const auto another_i1_basis : i1_bases) {
        for (const auto another_i1_quadrature : i1_quadratures) {
          for (const auto yet_another_i1_basis : i1_bases) {
            for (const auto yet_another_i1_quadrature : i1_quadratures) {
              valid_basis_and_quadratures.emplace_back(
                  std::array{i1_basis, another_i1_basis, yet_another_i1_basis},
                  std::array{i1_quadrature, another_i1_quadrature,
                             yet_another_i1_quadrature});
            }
          }
          valid_basis_and_quadratures.emplace_back(
              std::array{i1_basis, another_i1_basis, Basis::Fourier},
              std::array{i1_quadrature, another_i1_quadrature,
                         Quadrature::Equiangular});
          valid_basis_and_quadratures.emplace_back(
              std::array{i1_basis, another_i1_basis, Basis::Cartoon},
              std::array{i1_quadrature, another_i1_quadrature,
                         Quadrature::AxialSymmetry});
          valid_basis_and_quadratures.emplace_back(
              std::array{i1_basis, Basis::Fourier, another_i1_basis},
              std::array{i1_quadrature, Quadrature::Equiangular,
                         another_i1_quadrature});
          valid_basis_and_quadratures.emplace_back(
              std::array{Basis::Fourier, i1_basis, another_i1_basis},
              std::array{Quadrature::Equiangular, i1_quadrature,
                         another_i1_quadrature});
        }
      }
      valid_basis_and_quadratures.emplace_back(
          std::array{i1_basis, Basis::Fourier, Basis::Fourier},
          std::array{i1_quadrature, Quadrature::Equiangular,
                     Quadrature::Equiangular});
      valid_basis_and_quadratures.emplace_back(
          std::array{Basis::Fourier, i1_basis, Basis::Fourier},
          std::array{Quadrature::Equiangular, i1_quadrature,
                     Quadrature::Equiangular});
      valid_basis_and_quadratures.emplace_back(
          std::array{Basis::Fourier, Basis::Fourier, i1_basis},
          std::array{Quadrature::Equiangular, Quadrature::Equiangular,
                     i1_quadrature});
      valid_basis_and_quadratures.emplace_back(
          std::array{i1_basis, Basis::SphericalHarmonic,
                     Basis::SphericalHarmonic},
          std::array{i1_quadrature, Quadrature::Gauss,
                     Quadrature::Equiangular});
      valid_basis_and_quadratures.emplace_back(
          std::array{i1_basis, Basis::Cartoon, Basis::Cartoon},
          std::array{i1_quadrature, Quadrature::SphericalSymmetry,
                     Quadrature::SphericalSymmetry});
      valid_basis_and_quadratures.emplace_back(
          std::array{Basis::ZernikeB2, Basis::ZernikeB2, i1_basis},
          std::array{Quadrature::GaussRadauUpper, Quadrature::Equiangular,
                     i1_quadrature});
      valid_basis_and_quadratures.emplace_back(
          std::array{Basis::ZernikeB1, i1_basis, Basis::Cartoon},
          std::array{Quadrature::GaussRadauUpper, i1_quadrature,
                     Quadrature::AxialSymmetry});
    }
  }

  CHECK(valid_basis_and_quadratures.size() == 159);

  std::vector<std::array<Basis, 3>> bases;
  bases.reserve(729);
  for (const auto xi_basis : all_bases()) {
    if (xi_basis == Basis::Uninitialized) {
      continue;
    }
    for (const auto eta_basis : all_bases()) {
      if (eta_basis == Basis::Uninitialized) {
        continue;
      }
      for (const auto zeta_basis : all_bases()) {
        if (zeta_basis == Basis::Uninitialized) {
          continue;
        }
        bases.emplace_back(std::array{xi_basis, eta_basis, zeta_basis});
      }
    }
  }
  CHECK(bases.size() == 729);

  std::vector<std::array<Quadrature, 3>> quadratures;
  quadratures.reserve(729);
  for (const auto xi_quadrature : all_quadratures()) {
    if (xi_quadrature == Quadrature::Uninitialized) {
      continue;
    }
    for (const auto eta_quadrature : all_quadratures()) {
      if (eta_quadrature == Quadrature::Uninitialized) {
        continue;
      }
      for (const auto zeta_quadrature : all_quadratures()) {
        if (zeta_quadrature == Quadrature::Uninitialized) {
          continue;
        }
        quadratures.emplace_back(
            std::array{xi_quadrature, eta_quadrature, zeta_quadrature});
      }
    }
  }
  CHECK(quadratures.size() == 729);

  for (const auto basis : bases) {
    CAPTURE(basis);
    for (const auto quadrature : quadratures) {
      CAPTURE(quadrature);
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<3>(std::array{low_extent(basis[0], quadrature[0]), 5_st, 5_st},
                  basis, quadrature)));
      CHECK_FALSE(is_valid_dg_mesh(
          Mesh<3>(std::array{5_st, 5_st, high_extent(basis[1], quadrature[1])},
                  basis, quadrature)));
      if (alg::found(valid_basis_and_quadratures,
                     std::pair{basis, quadrature})) {
        if (basis[1] == Basis::SphericalHarmonic) {
          const size_t nth = random_extent(generator, basis[1], quadrature[1]);
          CHECK(is_valid_dg_mesh(Mesh<3>(
              std::array{random_extent(generator, basis[0], quadrature[0]), nth,
                         2 * nth - 1},
              basis, quadrature)));
        } else if (basis == bases::full_sphere) {
          const size_t nr = random_extent(generator, basis[0], quadrature[0]);
          CHECK(is_valid_dg_mesh(Mesh<3>(std::array{nr, 2 * nr - 1, 4 * nr - 3},
                                         basis, quadrature)));
        } else if (basis[0] == Basis::ZernikeB2) {
          const size_t nr = random_extent(generator, basis[0], quadrature[0]);
          CHECK(is_valid_dg_mesh(Mesh<3>(
              std::array{nr, 4 * nr - 3,
                         random_extent(generator, basis[2], quadrature[2])},
              basis, quadrature)));
        } else if (basis[2] == Basis::Cartoon) {
          if (basis[1] == Basis::Cartoon) {
            CHECK(is_valid_dg_mesh(Mesh<3>(
                std::array{random_extent(generator, basis[0], quadrature[0]),
                           1_st, 1_st},
                basis, quadrature)));
          } else {
            CHECK(is_valid_dg_mesh(Mesh<3>(
                std::array{random_extent(generator, basis[0], quadrature[0]),
                           random_extent(generator, basis[1], quadrature[1]),
                           1_st},
                basis, quadrature)));
          }
        } else {
          CHECK(is_valid_dg_mesh(Mesh<3>(
              std::array{random_extent(generator, basis[0], quadrature[0]),
                         random_extent(generator, basis[1], quadrature[1]),
                         random_extent(generator, basis[2], quadrature[2])},
              basis, quadrature)));
        }
      } else {
        CHECK_FALSE(is_valid_dg_mesh(Mesh<3>(
            std::array{random_extent(generator, basis[0], quadrature[0]),
                       random_extent(generator, basis[1], quadrature[1]),
                       random_extent(generator, basis[2], quadrature[2])},
            basis, quadrature)));
      }
    }
  }
  CHECK_FALSE(
      is_valid_dg_mesh(Mesh<3>{4_st, Basis::Fourier, Quadrature::Equiangular}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{{4_st, 4_st, 9_st},
              bases::spherical_shell<Basis::Legendre>,
              quadratures::spherical_shell<Quadrature::GaussLobatto>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{{4_st, 9_st, 5_st},
              bases::full_cylinder<Basis::Legendre>,
              quadratures::full_cylinder<Quadrature::GaussLobatto>}));
  CHECK_FALSE(is_valid_dg_mesh(Mesh<3>{
      {4_st, 5_st, 9_st}, bases::full_sphere, quadratures::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(Mesh<3>{
      {4_st, 7_st, 15_st}, bases::full_sphere, quadratures::full_sphere}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{{4_st, 4_st, 4_st},
              bases::cartoon_cylinder<Basis::Legendre>,
              quadratures::cartoon_cylinder<Quadrature::GaussLobatto>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{{4_st, 4_st, 4_st},
              bases::cartoon_cylinder_inner<Basis::Legendre>,
              quadratures::cartoon_cylinder_inner<Quadrature::GaussLobatto>}));
  CHECK_FALSE(is_valid_dg_mesh(
      Mesh<3>{{4_st, 4_st, 1_st},
              bases::cartoon_sphere<Basis::Legendre>,
              quadratures::cartoon_sphere<Quadrature::GaussLobatto>}));
  CHECK_FALSE(is_valid_dg_mesh(Mesh<3>{{4_st, 4_st, 1_st},
                                       bases::cartoon_sphere_inner,
                                       quadratures::cartoon_sphere_inner}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Spectral.IsValidDgMesh",
                  "[NumericalAlgorithms][Unit]") {
  MAKE_GENERATOR(generator);
  test_1d(make_not_null(&generator));
  test_2d(make_not_null(&generator));
  test_3d(make_not_null(&generator));
}
}  // namespace Spectral
