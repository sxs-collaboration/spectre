// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/GramSchmidtOrthonormalize.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.Tensor.EagerMath.GramSchmidtOrthonormalize",
                  "[DataStructures][Unit]") {
  const tnsr::aa<double, 3> minkowski_metric = []() {
    tnsr::aa<double, 3> metric{};
    get<0, 0>(metric) = -1.0;
    get<1, 1>(metric) = 1.0;
    get<2, 2>(metric) = 1.0;
    get<3, 3>(metric) = 1.0;
    return metric;
  }();
  const tnsr::ii<double, 3> flat_spatial = []() {
    tnsr::ii<double, 3> metric{};
    get<0, 0>(metric) = 1.0;
    get<1, 1>(metric) = 1.0;
    get<2, 2>(metric) = 1.0;
    return metric;
  }();

  {
    INFO("Already orthonormal basis");
    tnsr::A<double, 3> vec1{{{1.0, 0.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec2{{{0.0, 1.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec3{{{0.0, 0.0, 1.0, 0.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        minkowski_metric);

    CHECK(get(dot_product(vec1, vec1, minkowski_metric)) == approx(-1.0));
    CHECK(get(dot_product(vec2, vec2, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, minkowski_metric)) == approx(0.0));
  }

  {
    INFO("Spacetime vectors");
    tnsr::A<double, 3> vec1{{{2.0, 1.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec2{{{1.0, 1.0, 0.0, 0.0}}};  // <- null vector!
    tnsr::A<double, 3> vec3{{{1.0, 2.0, -1.0, 1.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        minkowski_metric);

    CHECK(get(dot_product(vec1, vec1, minkowski_metric)) == approx(-1.0));
    // Null vector is now spatial
    CHECK(get(dot_product(vec2, vec2, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, minkowski_metric)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, minkowski_metric)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, minkowski_metric)) == approx(0.0));
  }

  {
    INFO("Spatial vectors");
    tnsr::I<double, 3> vec1{{{1.0, 0.0, 0.0}}};
    tnsr::I<double, 3> vec2{{{1.0, 1.0, 0.0}}};
    tnsr::I<double, 3> vec3{{{1.0, 1.0, 1.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        flat_spatial);

    CHECK(get(dot_product(vec1, vec1, flat_spatial)) == approx(1.0));
    CHECK(get(dot_product(vec2, vec2, flat_spatial)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, flat_spatial)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, flat_spatial)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, flat_spatial)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, flat_spatial)) == approx(0.0));
  }

  {
    INFO("Nontrivial metric");
    const tnsr::aa<double, 3> tilted_metric = []() {
      tnsr::aa<double, 3> metric{};
      get<0, 0>(metric) = -2.0;
      get<0, 1>(metric) = -0.5;
      get<0, 2>(metric) = 0.3;
      get<0, 3>(metric) = -0.1;
      get<1, 1>(metric) = 3.0;
      get<1, 2>(metric) = 0.4;
      get<1, 3>(metric) = 0.2;
      get<2, 2>(metric) = 1.5;
      get<2, 3>(metric) = -0.3;
      get<3, 3>(metric) = 2.5;
      return metric;
    }();

    tnsr::A<double, 3> vec1{{{2.0, 1.0, 0.0, 0.0}}};
    tnsr::A<double, 3> vec2{{{1.0, 2.0, 1.0, 0.0}}};
    tnsr::A<double, 3> vec3{{{0.5, 0.0, 1.0, -1.0}}};

    gram_schmidt_orthonormalize(
        std::array{make_not_null(&vec1), make_not_null(&vec2),
                   make_not_null(&vec3)},
        tilted_metric);

    CHECK(get(dot_product(vec1, vec1, tilted_metric)) == approx(-1.0));
    CHECK(get(dot_product(vec2, vec2, tilted_metric)) == approx(1.0));
    CHECK(get(dot_product(vec3, vec3, tilted_metric)) == approx(1.0));
    CHECK(get(dot_product(vec1, vec2, tilted_metric)) == approx(0.0));
    CHECK(get(dot_product(vec1, vec3, tilted_metric)) == approx(0.0));
    CHECK(get(dot_product(vec2, vec3, tilted_metric)) == approx(0.0));
  }
}
