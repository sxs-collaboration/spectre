// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

template <typename DataType>
void test_contractions_rank2(const DataType& used_for_size) noexcept {
  // Contract (upper, lower) tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rul(used_for_size);
  create_tensor(make_not_null(&Rul));

  const auto RIi_expr = Rul(ti_I, ti_i);
  const std::array<size_t, 2> expected_multi_index{{0, 0}};
  CHECK(RIi_expr.get_first_uncontracted_lhs_multi_index_to_sum({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RIi_contracted = TensorExpressions::evaluate(RIi_expr);

  DataType expected_RIi_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    expected_RIi_sum += Rul.get(i, i);
  }
  CHECK(RIi_contracted.get() == expected_RIi_sum);

  // Contract (lower, upper) tensor
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Rlu(used_for_size);
  create_tensor(make_not_null(&Rlu));

  const auto RgG_expr = Rlu(ti_g, ti_G);
  CHECK(RgG_expr.get_first_uncontracted_lhs_multi_index_to_sum({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RgG_contracted = TensorExpressions::evaluate(RgG_expr);

  DataType expected_RgG_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t g = 0; g < 4; g++) {
    expected_RgG_sum += Rlu.get(g, g);
  }
  CHECK(RgG_contracted.get() == expected_RgG_sum);
}

template <typename DataType>
void test_contractions_rank3(const DataType& used_for_size) noexcept {
  // Contract first and second indices of (lower, upper, lower) tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rlul(used_for_size);
  create_tensor(make_not_null(&Rlul));

  const auto RiIj_expr = Rlul(ti_i, ti_I, ti_j);
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      RiIj_contracted = TensorExpressions::evaluate<ti_j>(RiIj_expr);

  for (size_t j = 0; j < 4; j++) {
    const std::array<size_t, 3> expected_multi_index{{0, 0, j}};
    CHECK(RiIj_expr.get_first_uncontracted_lhs_multi_index_to_sum({{j}}) ==
          expected_multi_index);

    DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Rlul.get(i, i, j);
    }
    CHECK(RiIj_contracted.get(j) == expected_sum);
  }

  // Contract first and third indices of (upper, upper, lower) tensor
  Tensor<DataType, Symmetry<2, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Ruul(used_for_size);
  create_tensor(make_not_null(&Ruul));

  const auto RJLj_expr = Ruul(ti_J, ti_L, ti_j);
  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RJLj_contracted = TensorExpressions::evaluate<ti_L>(RJLj_expr);

  for (size_t l = 0; l < 3; l++) {
    const std::array<size_t, 3> expected_multi_index{{0, l, 0}};
    CHECK(RJLj_expr.get_first_uncontracted_lhs_multi_index_to_sum({{l}}) ==
          expected_multi_index);

    DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t j = 0; j < 3; j++) {
      expected_sum += Ruul.get(j, l, j);
    }
    CHECK(RJLj_contracted.get(l) == expected_sum);
  }

  // Contract second and third indices of (upper, lower, upper) tensor
  Tensor<DataType, Symmetry<2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rulu(used_for_size);
  create_tensor(make_not_null(&Rulu));

  const auto RBfF_expr = Rulu(ti_B, ti_f, ti_F);
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      RBfF_contracted = TensorExpressions::evaluate<ti_B>(RBfF_expr);

  for (size_t b = 0; b < 4; b++) {
    const std::array<size_t, 3> expected_multi_index{{b, 0, 0}};
    CHECK(RBfF_expr.get_first_uncontracted_lhs_multi_index_to_sum({{b}}) ==
          expected_multi_index);

    DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t f = 0; f < 4; f++) {
      expected_sum += Rulu.get(b, f, f);
    }
    CHECK(RBfF_contracted.get(b) == expected_sum);
  }

  // Contract first and third indices of (lower, lower, upper) tensor with mixed
  // TensorIndexTypes
  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      Rllu(used_for_size);
  create_tensor(make_not_null(&Rllu));

  const auto RiaI_expr = Rllu(ti_i, ti_a, ti_I);
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RiaI_contracted = TensorExpressions::evaluate<ti_a>(RiaI_expr);

  for (size_t a = 0; a < 4; a++) {
    const std::array<size_t, 3> expected_multi_index{{0, a, 0}};
    CHECK(RiaI_expr.get_first_uncontracted_lhs_multi_index_to_sum({{a}}) ==
          expected_multi_index);

    DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Rllu.get(i, a, i);
    }
    CHECK(RiaI_contracted.get(a) == expected_sum);
  }
}

template <typename DataType>
void test_contractions_rank4(const DataType& used_for_size) noexcept {
  // Contract first and second indices of (lower, upper, upper, lower) tensor to
  // rank 2 tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rluul(used_for_size);
  create_tensor(make_not_null(&Rluul));

  const auto RiIKj_expr = Rluul(ti_i, ti_I, ti_K, ti_j);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      RiIKj_contracted =
          TensorExpressions::evaluate<ti_K, ti_j>(RiIKj_expr);

  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < 3; j++) {
      const std::array<size_t, 4> expected_multi_index{{0, 0, k, j}};
      CHECK(RiIKj_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{k, j}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t i = 0; i < 3; i++) {
        expected_sum += Rluul.get(i, i, k, j);
      }
      CHECK(RiIKj_contracted.get(k, j) == expected_sum);
    }
  }

  // Contract first and third indices of (upper, upper, lower, lower) tensor to
  // rank 2 tensor
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      Ruull(used_for_size);
  create_tensor(make_not_null(&Ruull));

  const auto RABac_expr = Ruull(ti_A, ti_B, ti_a, ti_c);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      RABac_contracted =
          TensorExpressions::evaluate<ti_B, ti_c>(RABac_expr);

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 5; c++) {
      const std::array<size_t, 4> expected_multi_index{{0, b, 0, c}};
      CHECK(RABac_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{b, c}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t a = 0; a < 5; a++) {
        expected_sum += Ruull.get(a, b, a, c);
      }
      CHECK(RABac_contracted.get(b, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of (upper, upper, upper, lower) tensor to
  // rank 2 tensor
  Tensor<DataType, Symmetry<3, 2, 3, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Ruuul(used_for_size);
  create_tensor(make_not_null(&Ruuul));

  const auto RLJIl_expr = Ruuul(ti_L, ti_J, ti_I, ti_l);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RLJIl_contracted =
          TensorExpressions::evaluate<ti_J, ti_I>(RLJIl_expr);

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 3; i++) {
      const std::array<size_t, 4> expected_multi_index{{0, j, i, 0}};
      CHECK(RLJIl_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{j, i}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t l = 0; l < 3; l++) {
        expected_sum += Ruuul.get(l, j, i, l);
      }
      CHECK(RLJIl_contracted.get(j, i) == expected_sum);
    }
  }

  // Contract second and third indices of (upper, upper, lower, upper) tensor to
  // rank 2 tensor
  Tensor<DataType, Symmetry<2, 2, 1, 2>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ruulu(used_for_size);
  create_tensor(make_not_null(&Ruulu));

  const auto REDdA_expr = Ruulu(ti_E, ti_D, ti_d, ti_A);
  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      REDdA_contracted =
          TensorExpressions::evaluate<ti_E, ti_A>(REDdA_expr);

  for (size_t e = 0; e < 4; e++) {
    for (size_t a = 0; a < 4; a++) {
      const std::array<size_t, 4> expected_multi_index{{e, 0, 0, a}};
      CHECK(REDdA_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{e, a}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t d = 0; d < 4; d++) {
        expected_sum += Ruulu.get(e, d, d, a);
      }
      CHECK(REDdA_contracted.get(e, a) == expected_sum);
    }
  }

  // Contract second and fourth indices of (lower, upper, lower, lower) tensor
  // to rank 2 tensor
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rlull(used_for_size);
  create_tensor(make_not_null(&Rlull));

  const auto RkJij_expr = Rlull(ti_k, ti_J, ti_i, ti_j);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      RkJij_contracted =
          TensorExpressions::evaluate<ti_k, ti_i>(RkJij_expr);

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 4; i++) {
      const std::array<size_t, 4> expected_multi_index{{k, 0, i, 0}};
      CHECK(RkJij_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{k, i}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Rlull.get(k, j, i, j);
      }
      CHECK(RkJij_contracted.get(k, i) == expected_sum);
    }
  }

  // Contract third and fourth indices of (upper, lower, lower, upper) tensor to
  // rank 2 tensor
  Tensor<DataType, Symmetry<3, 2, 2, 1>,
         index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rullu(used_for_size);
  create_tensor(make_not_null(&Rullu));

  const auto RFcgG_expr = Rullu(ti_F, ti_c, ti_g, ti_G);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      RFcgG_contracted =
          TensorExpressions::evaluate<ti_F, ti_c>(RFcgG_expr);

  for (size_t f = 0; f < 5; f++) {
    for (size_t c = 0; c < 4; c++) {
      const std::array<size_t, 4> expected_multi_index{{f, c, 0, 0}};
      CHECK(RFcgG_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{f, c}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t g = 0; g < 4; g++) {
        expected_sum += Rullu.get(f, c, g, g);
      }
      CHECK(RFcgG_contracted.get(f, c) == expected_sum);
    }
  }

  // Contract first and second indices of (upper, lower, upper, upper) tensor to
  // rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<3, 2, 3, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<2, UpLo::Up, Frame::Grid>>>
      Ruluu(used_for_size);
  create_tensor(make_not_null(&Ruluu));

  const auto RKkIJ_expr = Ruluu(ti_K, ti_k, ti_I, ti_J);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RKkIJ_contracted_to_JI =
          TensorExpressions::evaluate<ti_J, ti_I>(RKkIJ_expr);

  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < 3; i++) {
      const std::array<size_t, 4> expected_multi_index{{0, 0, j, i}};
      CHECK(RKkIJ_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{j, i}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t k = 0; k < 3; k++) {
        expected_sum += Ruluu.get(k, k, i, j);
      }
      CHECK(RKkIJ_contracted_to_JI.get(j, i) == expected_sum);
    }
  }

  // Contract first and third indices of (lower, upper, upper, upper) tensor to
  // rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<3, 2, 1, 2>,
         index_list<SpacetimeIndex<2, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>
      Rluuu(used_for_size);
  create_tensor(make_not_null(&Rluuu));

  const auto RbCBE_expr = Rluuu(ti_b, ti_C, ti_B, ti_E);
  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>
      RbCBE_contracted_to_EC =
          TensorExpressions::evaluate<ti_E, ti_C>(RbCBE_expr);

  for (size_t e = 0; e < 3; e++) {
    for (size_t c = 0; c < 3; c++) {
      const std::array<size_t, 4> expected_multi_index{{0, e, 0, c}};
      CHECK(RbCBE_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{e, c}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t b = 0; b < 3; b++) {
        expected_sum += Rluuu.get(b, c, b, e);
      }
      CHECK(RbCBE_contracted_to_EC.get(e, c) == expected_sum);
    }
  }

  // Contract first and fourth indices of (upper, lower, lower, lower) tensor to
  // rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<2, 1, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Rulll(used_for_size);
  create_tensor(make_not_null(&Rulll));

  const auto RAdba_expr = Rulll(ti_A, ti_d, ti_b, ti_a);
  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RAdba_contracted_to_bd =
          TensorExpressions::evaluate<ti_b, ti_d>(RAdba_expr);

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
      const std::array<size_t, 4> expected_multi_index{{0, b, d, 0}};
      CHECK(RAdba_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{b, d}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t a = 0; a < 4; a++) {
        expected_sum += Rulll.get(a, d, b, a);
      }
      CHECK(RAdba_contracted_to_bd.get(b, d) == expected_sum);
    }
  }

  // Contract second and third indices of (lower, lower, upper, lower) tensor to
  // rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rllul(used_for_size);
  create_tensor(make_not_null(&Rllul));

  const auto RljJi_expr = Rllul(ti_l, ti_j, ti_J, ti_i);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      RljJi_contracted_to_il =
          TensorExpressions::evaluate<ti_i, ti_l>(RljJi_expr);

  for (size_t i = 0; i < 4; i++) {
    for (size_t l = 0; l < 3; l++) {
      const std::array<size_t, 4> expected_multi_index{{i, 0, 0, l}};
      CHECK(RljJi_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{i, l}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_sum += Rllul.get(l, j, j, i);
      }
      CHECK(RljJi_contracted_to_il.get(i, l) == expected_sum);
    }
  }

  // Contract second and fourth indices of (lower, lower, upper, upper) tensor
  // to rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<2, 2, 1, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Rlluu(used_for_size);
  create_tensor(make_not_null(&Rlluu));

  const auto RagDG_expr = Rlluu(ti_a, ti_g, ti_D, ti_G);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      RagDG_contracted_to_Da =
          TensorExpressions::evaluate<ti_D, ti_a>(RagDG_expr);

  for (size_t d = 0; d < 4; d++) {
    for (size_t a = 0; a < 4; a++) {
      const std::array<size_t, 4> expected_multi_index{{d, 0, a, 0}};
      CHECK(RagDG_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{d, a}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t g = 0; g < 4; g++) {
        expected_sum += Rlluu.get(a, g, d, g);
      }
      CHECK(RagDG_contracted_to_Da.get(d, a) == expected_sum);
    }
  }

  // Contract third and fourth indices of (lower, upper, lower, upper) tensor to
  // rank 2 tensor and reorder indices
  Tensor<DataType, Symmetry<2, 1, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Rlulu(used_for_size);
  create_tensor(make_not_null(&Rlulu));

  const auto RlJiI_expr = Rlulu(ti_l, ti_J, ti_i, ti_I);
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      RlJiI_contracted_to_Jl =
          TensorExpressions::evaluate<ti_J, ti_l>(RlJiI_expr);

  for (size_t j = 0; j < 3; j++) {
    for (size_t l = 0; l < 3; l++) {
      const std::array<size_t, 4> expected_multi_index{{j, l, 0, 0}};
      CHECK(RlJiI_expr.get_first_uncontracted_lhs_multi_index_to_sum(
                {{j, l}}) == expected_multi_index);

      DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t i = 0; i < 3; i++) {
        expected_sum += Rlulu.get(l, j, i, i);
      }
      CHECK(RlJiI_contracted_to_Jl.get(j, l) == expected_sum);
    }
  }

  // Contract first and second indices AND third and fourth indices to rank 0
  // tensor
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rulul(used_for_size);
  create_tensor(make_not_null(&Rulul));

  const auto RKkLl_expr = Rulul(ti_K, ti_k, ti_L, ti_l);
  // `RKkLl_expr` is a TensorContract expression that contains another
  // TensorContract expression. The "inner" expression will contract the L/l
  // indices, representing contracting the 3rd and 4th indices of the rank 4
  // tensor `Rulul` to a rank 2 tensor. The "outer" expression will then
  // contract the K/k indices, representing contracting the rank 2 tensor to
  // the resulting scalar. This `get_first_uncontracted_lhs_multi_index_to_sum`
  // test checks this outer contraction of the K/k indices. Because the inner
  // expression is private, a similar check for it is not done.
  //
  // This also applies to similar rank 4 -> rank 0 contraction cases below
  const std::array<size_t, 2> expected_multi_index{{0, 0}};
  CHECK(RKkLl_expr.get_first_uncontracted_lhs_multi_index_to_sum({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RKkLl_contracted =
      TensorExpressions::evaluate(RKkLl_expr);

  DataType expected_RKkLl_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 4; l++) {
      expected_RKkLl_sum += Rulul.get(k, k, l, l);
    }
  }
  CHECK(RKkLl_contracted.get() == expected_RKkLl_sum);

  // Contract first and third indices and second and fourth indices to rank 0
  // tensor
  const auto RcaCA_expr = Rlluu(ti_c, ti_a, ti_C, ti_A);
  CHECK(RcaCA_expr.get_first_uncontracted_lhs_multi_index_to_sum({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RcaCA_contracted =
      TensorExpressions::evaluate(RcaCA_expr);

  DataType expected_RcaCA_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_RcaCA_sum += Rlluu.get(c, a, c, a);
    }
  }
  CHECK(RcaCA_contracted.get() == expected_RcaCA_sum);

  // Contract first and fourth indices and second and third indices to rank 0
  // tensor
  const auto RjIiJ_expr = Rlulu(ti_j, ti_I, ti_i, ti_J);
  CHECK(RjIiJ_expr.get_first_uncontracted_lhs_multi_index_to_sum({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RjIiJ_contracted =
      TensorExpressions::evaluate(RjIiJ_expr);

  DataType expected_RjIiJ_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_RjIiJ_sum += Rlulu.get(j, i, i, j);
    }
  }
  CHECK(RjIiJ_contracted.get() == expected_RjIiJ_sum);
}

template <typename DataType>
void test_spatial_spacetime_index(const DataType& used_for_size) noexcept {
  // Contract (spatial, spacetime) tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>

      R(used_for_size);
  create_tensor(make_not_null(&R));
  const Tensor<DataType> R_contracted =
      TensorExpressions::evaluate(R(ti_I, ti_i));

  // Contract (spacetime, spatial) tensor
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      S(used_for_size);
  create_tensor(make_not_null(&S));
  const Tensor<DataType> S_contracted =
      TensorExpressions::evaluate(S(ti_K, ti_k));

  // Contract (spacetime, spacetime) tensor using generic spatial indices
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      T(used_for_size);
  create_tensor(make_not_null(&T));
  const Tensor<DataType> T_contracted =
      TensorExpressions::evaluate(T(ti_j, ti_J));

  DataType expected_R_sum = make_with_value<DataType>(used_for_size, 0.0);
  DataType expected_S_sum = make_with_value<DataType>(used_for_size, 0.0);
  DataType expected_T_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    expected_R_sum += R.get(i, i + 1);
    expected_S_sum += S.get(i + 1, i);
    expected_T_sum += T.get(i + 1, i + 1);
  }
  CHECK_ITERABLE_APPROX(R_contracted.get(), expected_R_sum);
  CHECK_ITERABLE_APPROX(S_contracted.get(), expected_S_sum);
  CHECK_ITERABLE_APPROX(T_contracted.get(), expected_T_sum);

  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      G(used_for_size);
  create_tensor(make_not_null(&G));

  // Contract one (spatial, spacetime) pair of indices of a tensor that also
  // takes a generic spatial index for a single non-contracted spacetime index
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      G_contracted_1 =
          TensorExpressions::evaluate<ti_i, ti_K>(G(ti_K, ti_j, ti_i, ti_J));

  for (size_t i = 0; i < 3; i++) {
    for (size_t k = 0; k < 3; k++) {
      DataType expected_G_sum_1 = make_with_value<DataType>(used_for_size, 0.0);
      for (size_t j = 0; j < 3; j++) {
        expected_G_sum_1 += G.get(k, j, i + 1, j + 1);
      }
      CHECK_ITERABLE_APPROX(G_contracted_1.get(i, k), expected_G_sum_1);
    }
  }

  // Contract one (spacetime, spacetime) pair of indices using generic spatial
  // indices and then one (spatial, spatial) pair of indices
  const Tensor<DataType> G_contracted_2 =
      TensorExpressions::evaluate(G(ti_I, ti_i, ti_j, ti_J));

  DataType expected_G_sum_2 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_G_sum_2 += G.get(i, i, j + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(G_contracted_2.get(), expected_G_sum_2);

  // Contract two (spatial, spacetime) pairs of indices
  const Tensor<DataType> G_contracted_3 =
      TensorExpressions::evaluate(G(ti_I, ti_j, ti_i, ti_J));

  DataType expected_G_sum_3 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      expected_G_sum_3 += G.get(i, j, i + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(G_contracted_3.get(), expected_G_sum_3);

  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      H(used_for_size);
  create_tensor(make_not_null(&H));

  // Contract one (spacetime, spacetime) pair of indices using generic spacetime
  // indices and then one (spacetime, spacetime) pair of indices using generic
  // spatial indices
  const Tensor<DataType> H_contracted_1 =
      TensorExpressions::evaluate(H(ti_i, ti_I, ti_a, ti_A));

  DataType expected_H_sum_1 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t i = 0; i < 3; i++) {
    for (size_t a = 0; a < 4; a++) {
      expected_H_sum_1 += H.get(i + 1, i + 1, a, a);
    }
  }
  CHECK_ITERABLE_APPROX(H_contracted_1.get(), expected_H_sum_1);

  // Contract two (spacetime, spacetime) pair of indices using generic spatial
  // indices
  const Tensor<DataType> H_contracted_2 =
      TensorExpressions::evaluate(H(ti_j, ti_I, ti_i, ti_J));

  DataType expected_H_sum_2 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_H_sum_2 += H.get(j + 1, i + 1, i + 1, j + 1);
    }
  }
  CHECK_ITERABLE_APPROX(H_contracted_2.get(), expected_H_sum_2);
}

template <typename DataType>
void test_contractions(const DataType& used_for_size) noexcept {
  test_contractions_rank2(used_for_size);
  test_contractions_rank3(used_for_size);
  test_contractions_rank4(used_for_size);
  test_spatial_spacetime_index(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  test_contractions(std::numeric_limits<double>::signaling_NaN());
  test_contractions(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
