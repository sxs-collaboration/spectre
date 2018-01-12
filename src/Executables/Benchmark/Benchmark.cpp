// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <benchmark/benchmark.h>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Element.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// static void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// static void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// static void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
// In this anonymous namespace is an example of microbenchmarking the
// all_gradient routine for the GH system

template <size_t Dim>
struct Kappa : db::DataBoxTag {
  using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString label = "Kappa";
};
template <size_t Dim>
struct Psi : db::DataBoxTag {
  using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString label = "Psi";
};

static void bench_all_gradient(benchmark::State& state) {
  constexpr const size_t pts_1d = 4;
  constexpr const size_t Dim = 3;
  const Index<Dim> extents(pts_1d);
  CoordinateMaps::AffineMap map1d(-1.0, 1.0, -1.0, 1.0);
  using Map3d = CoordinateMaps::ProductOf3Maps<CoordinateMaps::AffineMap,
                                               CoordinateMaps::AffineMap,
                                               CoordinateMaps::AffineMap>;
  CoordinateMap<Frame::Logical, Frame::Grid, Map3d> map(
      Map3d{map1d, map1d, map1d});

  using VarTags = typelist<Kappa<Dim>, Psi<Dim>>;
  const InverseJacobian<Dim, Frame::Logical, Frame::Grid> inv_jac =
      map.inv_jacobian(logical_coordinates(extents));
  const auto grid_coords = map(logical_coordinates(extents));
  Variables<VarTags> vars(extents.product(), 0.0);

  while (state.KeepRunning()) {
    benchmark::DoNotOptimize(
        partial_derivatives<VarTags>(vars, extents, inv_jac));
  }
}
BENCHMARK(bench_all_gradient);
}  // namespace

BENCHMARK_MAIN()
