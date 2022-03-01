// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<double, Ts...>*> tensor) {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    const gsl::not_null<Tensor<DataVector, Ts...>*> tensor) {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

// Checks that the number of ops in the expressions match what is expected
void test_tensor_ops_properties() {
  const Scalar<double> G{};
  const double H = 0.0;
  const tnsr::ii<double, 3> R{};
  const tnsr::ij<double, 3> S{};
  const tnsr::aa<double, 3> T{};

  const auto scalar_expression = H - G() - H;
  const auto R_plus_S = R(ti::i, ti::j) + S(ti::i, ti::j);
  const auto R_minus_T = R(ti::i, ti::j) - T(ti::j, ti::i);
  const auto large_expression = R(ti::i, ti::j) + S(ti::i, ti::j) -
                                T(ti::j, ti::i) + S(ti::j, ti::i) +
                                S(ti::i, ti::j) - R(ti::j, ti::i);

  CHECK(scalar_expression.num_ops_subtree == 2);
  CHECK(R_plus_S.num_ops_subtree == 1);
  CHECK(R_minus_T.num_ops_subtree == 1);
  CHECK(large_expression.num_ops_subtree == 5);
}

// \brief Test the sum and difference of a `double` and tensor expression is
// correctly evaluated
//
// \details
// The cases tested are:
// - \f$L = R + S\f$
// - \f$L = R - S\f$
// - \f$L = G^{i}{}_{i} + R\f$
// - \f$L = G^{i}{}_{i} - R\f$
// - \f$L = R + S + T\f$
// - \f$L = R - G^{i}{}_{i} + T\f$
//
// where \f$R\f$ and \f$T\f$ are `double`s and \f$S\f$, \f$G\f$, and \f$L\f$
// are Tensors with data type `double` or DataVector.
//
// \tparam DataType the type of data being stored in the tensor expression
// operand of the sums and differences
template <typename DataType>
void test_addsub_double(const DataType& used_for_size) {
  Tensor<DataType> S{{{used_for_size}}};
  if (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` to a proper test value
    S.get() = 2.4;
  } else {
    assign_unique_values_to_tensor(make_not_null(&S));
  }

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      G(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&G));

  DataType G_trace = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    G_trace += G.get(i, i);
  }

  // \f$L = R + S\f$
  const Tensor<DataType> R_plus_S = tenex::evaluate(5.6 + S());
  // \f$L = R - S\f$
  const Tensor<DataType> R_minus_S = tenex::evaluate(1.1 - S());
  // \f$L = G^{i}{}_{i} + R\f$
  const Tensor<DataType> G_plus_R = tenex::evaluate(G(ti::I, ti::i) + 8.2);
  // \f$L = G^{i}{}_{i} - R\f$
  const Tensor<DataType> G_minus_R = tenex::evaluate(G(ti::I, ti::i) - 3.5);
  // \f$L = R + S + T\f$
  const Tensor<DataType> R_plus_S_plus_T = tenex::evaluate(0.7 + S() + 9.8);
  // \f$L = R - G^{i}{}_{i} + T\f$
  const Tensor<DataType> R_minus_G_plus_T =
      tenex::evaluate(5.9 - G(ti::I, ti::i) + 4.7);

  CHECK(R_plus_S.get() == 5.6 + S.get());
  CHECK(R_minus_S.get() == 1.1 - S.get());
  CHECK(G_plus_R.get() == G_trace + 8.2);
  CHECK(G_minus_R.get() == G_trace - 3.5);
  CHECK(R_plus_S_plus_T.get() == 0.7 + S.get() + 9.8);
  CHECK(R_minus_G_plus_T.get() == 5.9 - G_trace + 4.7);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubtract",
                  "[DataStructures][Unit]") {
  test_tensor_ops_properties();

  // Test adding and subtracting `double`s
  test_addsub_double(std::numeric_limits<double>::signaling_NaN());
  test_addsub_double(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));

  // Test adding scalars
  const Tensor<double> scalar_1{{{2.1}}};
  const Tensor<double> scalar_2{{{-0.8}}};
  Tensor<double> lhs_scalar = tenex::evaluate(scalar_1() + scalar_2());
  CHECK(lhs_scalar.get() == 1.3);

  Tensor<double, Symmetry<1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      All{};
  std::iota(All.begin(), All.end(), 0.0);
  Tensor<double, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hll{};
  std::iota(Hll.begin(), Hll.end(), 0.0);
  // [use_tensor_index]
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll =
          tenex::evaluate<ti::a, ti::b>(All(ti::a, ti::b) + Hll(ti::a, ti::b));
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll2 =
          tenex::evaluate<ti::a, ti::b>(All(ti::a, ti::b) + Hll(ti::b, ti::a));
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll3 =
          tenex::evaluate<ti::a, ti::b>(All(ti::a, ti::b) + Hll(ti::b, ti::a) +
                                        All(ti::b, ti::a) - Hll(ti::b, ti::a));
  // [use_tensor_index]
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      CHECK(Gll.get(i, j) == All.get(i, j) + Hll.get(i, j));
      CHECK(Gll2.get(i, j) == All.get(i, j) + Hll.get(j, i));
      CHECK(Gll3.get(i, j) == 2.0 * All.get(i, j));
    }
  }
  // Test 3 indices add subtract
  Tensor<double, Symmetry<1, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Alll{};
  std::iota(Alll.begin(), Alll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 3>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Hlll{};
  std::iota(Hlll.begin(), Hlll.end(), 0.0);
  Tensor<double, Symmetry<2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rlll{};
  std::iota(Rlll.begin(), Rlll.end(), 0.0);
  Tensor<double, Symmetry<1, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Slll{};
  std::iota(Slll.begin(), Slll.end(), 0.0);

  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll = tenex::evaluate<ti::a, ti::b, ti::c>(Alll(ti::a, ti::b, ti::c) +
                                                  Hlll(ti::a, ti::b, ti::c));
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll2 = tenex::evaluate<ti::a, ti::b, ti::c>(Alll(ti::a, ti::b, ti::c) +
                                                   Hlll(ti::b, ti::a, ti::c));
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll3 = tenex::evaluate<ti::a, ti::b, ti::c>(
          Alll(ti::a, ti::b, ti::c) + Hlll(ti::b, ti::a, ti::c) +
          Alll(ti::b, ti::a, ti::c) - Hlll(ti::b, ti::a, ti::c));
  // testing LHS symmetry is nonsymmetric when RHS operands do not have
  // symmetries in common
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll4 = tenex::evaluate<ti::a, ti::b, ti::c>(Alll(ti::b, ti::c, ti::a) +
                                                   Rlll(ti::c, ti::a, ti::b));
  // testing LHS symmetry preserves shared RHS symmetry when RHS operands have
  // symmetries in common
  const Tensor<double, Symmetry<2, 1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll5 = tenex::evaluate<ti::a, ti::b, ti::c>(Alll(ti::b, ti::c, ti::a) -
                                                   Rlll(ti::a, ti::c, ti::b));

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        CHECK(Glll.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(i, j, k));
        CHECK(Glll2.get(i, j, k) == Alll.get(i, j, k) + Hlll.get(j, i, k));
        CHECK(Glll3.get(i, j, k) == 2.0 * Alll.get(i, j, k));
        CHECK(Glll4.get(i, j, k) == Alll.get(j, k, i) + Rlll.get(k, i, j));
        CHECK(Glll5.get(i, j, k) == Alll.get(j, k, i) - Rlll.get(i, k, j));
      }
    }
  }

  // testing with expressions having spatial indices for spacetime indices
  const Tensor<double, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll6 = tenex::evaluate<ti::a, ti::j, ti::k>(Rlll(ti::a, ti::j, ti::k) +
                                                   Slll(ti::a, ti::j, ti::k));
  Tensor<double, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Glll7{};
  tenex::evaluate<ti::j, ti::a, ti::k>(
      make_not_null(&Glll7),
      Slll(ti::j, ti::k, ti::a) - Rlll(ti::k, ti::a, ti::j));

  for (int a = 0; a < 4; ++a) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        CHECK(Glll6.get(a, j, k) ==
              Rlll.get(a, j + 1, k + 1) + Slll.get(a, j, k + 1));
        CHECK(Glll7.get(j, a, k + 1) ==
              Slll.get(j + 1, k, a) - Rlll.get(k + 1, a, j + 1));
      }
    }
  }

  // testing with operands having time indices for spacetime indices
  const Tensor<double, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll7 = tenex::evaluate<ti::c, ti::b>(Alll(ti::c, ti::t, ti::b) +
                                           Hlll(ti::b, ti::c, ti::t));

  for (int c = 0; c < 4; ++c) {
    for (int b = 0; b < 4; ++b) {
      CHECK(Gll7.get(c, b) == Alll.get(c, 0, b) + Hlll.get(b, c, 0));
    }
  }

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll8 = tenex::evaluate<ti::d, ti::c>(Alll(ti::c, ti::d, ti::t) -
                                           All(ti::d, ti::c));

  for (int d = 0; d < 4; ++d) {
    for (int c = 0; c < 4; ++c) {
      CHECK(Gll8.get(d, c) == Alll.get(c, d, 0) - All.get(d, c));
    }
  }

  const Tensor<double, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Gll9 = tenex::evaluate<ti::a, ti::b>(All(ti::b, ti::a) +
                                           Alll(ti::b, ti::a, ti::t));

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      CHECK(Gll9.get(a, b) == All.get(b, a) + Alll.get(b, a, 0));
    }
  }

  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  const double spatial_component_placeholder =
      std::numeric_limits<double>::max();

  auto Gll10 = make_with_value<
      Tensor<double, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>>(
      0.0, spatial_component_placeholder);
  tenex::evaluate<ti::t, ti::d, ti::b, ti::T>(
      make_not_null(&Gll10), All(ti::b, ti::d) - Hll(ti::b, ti::d));

  for (int d = 0; d < 4; ++d) {
    for (int b = 0; b < 4; ++b) {
      CHECK(Gll10.get(0, d, b, 0) == All.get(b, d) - Hll.get(b, d));
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          CHECK(Gll10.get(i + 1, d, b, j + 1) == spatial_component_placeholder);
        }
      }
    }
  }

  auto Gll11 = make_with_value<
      Tensor<double, Symmetry<2, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>>(
      0.0, spatial_component_placeholder);
  tenex::evaluate<ti::a, ti::c, ti::t>(
      make_not_null(&Gll11), Rlll(ti::t, ti::c, ti::a) + All(ti::a, ti::c));

  for (int a = 0; a < 4; ++a) {
    for (int c = 0; c < 4; ++c) {
      CHECK(Gll11.get(a, c, 0) == Rlll.get(0, c, a) + All.get(a, c));
      for (size_t i = 0; i < 3; ++i) {
        CHECK(Gll11.get(a, c, i + 1) == spatial_component_placeholder);
      }
    }
  }

  auto Gll12 = make_with_value<
      Tensor<double, Symmetry<3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>>(
      0.0, spatial_component_placeholder);
  tenex::evaluate<ti::g, ti::t, ti::h>(
      make_not_null(&Gll12), Hll(ti::h, ti::g) - Alll(ti::g, ti::t, ti::h));

  for (int g = 0; g < 4; ++g) {
    for (int h = 0; h < 4; ++h) {
      CHECK(Gll12.get(g, 0, h) == Hll.get(h, g) - Alll.get(g, 0, h));
      for (size_t i = 0; i < 3; ++i) {
        CHECK(Gll12.get(g, i + 1, h) == spatial_component_placeholder);
      }
    }
  }

  auto Gll13 = make_with_value<
      Tensor<double, Symmetry<2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>>(
      0.0, spatial_component_placeholder);
  tenex::evaluate<ti::t, ti::f>(make_not_null(&Gll13),
                                Rlll(ti::t, ti::t, ti::f) - Hll(ti::f, ti::t));

  for (int f = 0; f < 4; ++f) {
    CHECK(Gll13.get(0, f) == Rlll.get(0, 0, f) - Hll.get(f, 0));
    for (size_t i = 0; i < 3; ++i) {
      CHECK(Gll13.get(i + 1, f) == spatial_component_placeholder);
    }
  }

  const Tensor<double> T{{{-3.2}}};

  auto Gll14 = make_with_value<
      Tensor<double, Symmetry<2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>>(
      0.0, spatial_component_placeholder);
  tenex::evaluate<ti::T, ti::t>(
      make_not_null(&Gll14),
      Hll(ti::t, ti::t) - T() - Rlll(ti::t, ti::t, ti::t) + 9.1);

  CHECK(Gll14.get(0, 0) == Hll.get(0, 0) - get(T) - Rlll.get(0, 0, 0) + 9.1);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      CHECK(Gll14.get(i + 1, j + 1) == spatial_component_placeholder);
    }
  }
}
