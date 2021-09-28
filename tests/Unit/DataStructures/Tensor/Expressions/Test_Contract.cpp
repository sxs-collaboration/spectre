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
void create_tensor(gsl::not_null<Tensor<double, Ts...>*> tensor) {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void create_tensor(gsl::not_null<Tensor<DataVector, Ts...>*> tensor) {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

const size_t contracted_value_placeholder = std::numeric_limits<size_t>::max();

template <typename DataType>
void test_contractions_rank2(const DataType& used_for_size) {
  // Contract (upper, lower) tensor
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rul(used_for_size);
  create_tensor(make_not_null(&Rul));

  const auto RIi_expr = Rul(ti_I, ti_i);
  const std::array<size_t, 2> expected_multi_index{
      {contracted_value_placeholder, contracted_value_placeholder}};
  CHECK(RIi_expr.get_uncontracted_multi_index_with_uncontracted_values({{}}) ==
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
  CHECK(RgG_expr.get_uncontracted_multi_index_with_uncontracted_values({{}}) ==
        expected_multi_index);

  const Tensor<DataType> RgG_contracted = TensorExpressions::evaluate(RgG_expr);

  DataType expected_RgG_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t g = 0; g < 4; g++) {
    expected_RgG_sum += Rlu.get(g, g);
  }
  CHECK(RgG_contracted.get() == expected_RgG_sum);
}

template <typename DataType>
void test_contractions_rank3(const DataType& used_for_size) {
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
    const std::array<size_t, 3> expected_multi_index{
        {contracted_value_placeholder, contracted_value_placeholder, j}};
    CHECK(RiIj_expr.get_uncontracted_multi_index_with_uncontracted_values(
              {{j}}) == expected_multi_index);

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
    const std::array<size_t, 3> expected_multi_index{
        {contracted_value_placeholder, l, contracted_value_placeholder}};
    CHECK(RJLj_expr.get_uncontracted_multi_index_with_uncontracted_values(
              {{l}}) == expected_multi_index);

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
    const std::array<size_t, 3> expected_multi_index{
        {b, contracted_value_placeholder, contracted_value_placeholder}};
    CHECK(RBfF_expr.get_uncontracted_multi_index_with_uncontracted_values(
              {{b}}) == expected_multi_index);

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
    const std::array<size_t, 3> expected_multi_index{
        {contracted_value_placeholder, a, contracted_value_placeholder}};
    CHECK(RiaI_expr.get_uncontracted_multi_index_with_uncontracted_values(
              {{a}}) == expected_multi_index);

    DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t i = 0; i < 3; i++) {
      expected_sum += Rllu.get(i, a, i);
    }
    CHECK(RiaI_contracted.get(a) == expected_sum);
  }
}

template <typename DataType>
void test_contractions_rank4(const DataType& used_for_size) {
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, contracted_value_placeholder, k, j}};
      CHECK(RiIKj_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, b, contracted_value_placeholder, c}};
      CHECK(RABac_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, j, i, contracted_value_placeholder}};
      CHECK(RLJIl_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {e, contracted_value_placeholder, contracted_value_placeholder, a}};
      CHECK(REDdA_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {k, contracted_value_placeholder, i, contracted_value_placeholder}};
      CHECK(RkJij_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {f, c, contracted_value_placeholder, contracted_value_placeholder}};
      CHECK(RFcgG_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, contracted_value_placeholder, j, i}};
      CHECK(RKkIJ_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, e, contracted_value_placeholder, c}};
      CHECK(RbCBE_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {contracted_value_placeholder, b, d, contracted_value_placeholder}};
      CHECK(RAdba_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {i, contracted_value_placeholder, contracted_value_placeholder, l}};
      CHECK(RljJi_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {d, contracted_value_placeholder, a, contracted_value_placeholder}};
      CHECK(RagDG_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
      const std::array<size_t, 4> expected_multi_index{
          {j, l, contracted_value_placeholder, contracted_value_placeholder}};
      CHECK(RlJiI_expr.get_uncontracted_multi_index_with_uncontracted_values(
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
  // the resulting scalar. This
  // `get_uncontracted_multi_index_with_uncontracted_values` test checks this
  // outer contraction of the K/k indices. Because the inner expression is
  // private, a similar check for it is not done.
  //
  // This also applies to similar rank 4 -> rank 0 contraction cases below
  const std::array<size_t, 2> expected_multi_index{
      {contracted_value_placeholder, contracted_value_placeholder}};
  CHECK(RKkLl_expr.get_uncontracted_multi_index_with_uncontracted_values(
            {{}}) == expected_multi_index);

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
  CHECK(RcaCA_expr.get_uncontracted_multi_index_with_uncontracted_values(
            {{}}) == expected_multi_index);

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
  CHECK(RjIiJ_expr.get_uncontracted_multi_index_with_uncontracted_values(
            {{}}) == expected_multi_index);

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
void test_time_index(const DataType& used_for_size) {
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      R(used_for_size);
  create_tensor(make_not_null(&R));
  // Contract RHS tensor with time index to a LHS tensor without a time index
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  // \f$L_{b} = R^{at}{}_{ab}\f$
  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      R_contracted_1 =
          TensorExpressions::evaluate<ti_b>(R(ti_A, ti_T, ti_a, ti_b));

  for (size_t b = 0; b < 4; b++) {
    DataType expected_R_sum_1 = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t a = 0; a < 4; a++) {
      expected_R_sum_1 += R.get(a, 0, a, b);
    }
    CHECK_ITERABLE_APPROX(R_contracted_1.get(b), expected_R_sum_1);
  }

  // Contract RHS tensor with upper and lower time index to a LHS tensor without
  // time indices
  // \f$L = R^{at}{}_{at}\f$
  //
  // Makes sure that `TensorContract` does not get confused by the presence of
  // an upper and lower time index in the RHS tensor, which is different than
  // the presence of an upper and lower generic index
  const Tensor<DataType> R_contracted_2 =
      TensorExpressions::evaluate(R(ti_A, ti_T, ti_a, ti_t));
  DataType expected_R_sum_2 = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t a = 0; a < 4; a++) {
    expected_R_sum_2 += R.get(a, 0, a, 0);
  }
  CHECK_ITERABLE_APPROX(R_contracted_2.get(), expected_R_sum_2);

  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      S(used_for_size);
  create_tensor(make_not_null(&S));
  // Assign a placeholder to the LHS tensor's components before it is computed
  // so that when test expressions below only compute time components, we can
  // check that LHS spatial components haven't changed
  const double spatial_component_placeholder =
      std::numeric_limits<double>::max();
  auto S_contracted = make_with_value<
      Tensor<DataType, Symmetry<4, 3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>>(
      used_for_size, spatial_component_placeholder);

  // Contract a RHS tensor without time indices to a LHS tensor with time
  // indices
  // \f$L_{tt}{}^{bt} = R^{ba}{}_{a}\f$
  //
  // Makes sure that `TensorContract` does not get confused by the presence of
  // an upper and lower time index in the LHS tensor, which is different than
  // the presence of an upper and lower generic index. Also makes sure that
  // a contraction can be evaluated to a LHS tensor of higher rank than the
  // rank that results from contracting the RHS
  ::TensorExpressions::evaluate<ti_t, ti_t, ti_B, ti_T>(
      make_not_null(&S_contracted), S(ti_B, ti_A, ti_a));

  for (size_t b = 0; b < 4; b++) {
    DataType expected_S_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t a = 0; a < 4; a++) {
      expected_S_sum += S.get(b, a, a);
    }
    CHECK_ITERABLE_APPROX(S_contracted.get(0, 0, b, 0), expected_S_sum);

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        for (size_t k = 0; k < 3; k++) {
          CHECK(S_contracted.get(i + 1, j + 1, b, k + 1) ==
                spatial_component_placeholder);
        }
      }
    }
  }

  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                    SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      T(used_for_size);
  create_tensor(make_not_null(&T));
  auto T_contracted = make_with_value<
      Tensor<DataType, Symmetry<3, 2, 1>,
             index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                        SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>>(
      used_for_size, spatial_component_placeholder);

  // Contract a RHS tensor with time indices to a LHS tensor with time indices
  // \f$L^{tb}{}_{t} = R_{at}{}^{ab}\f$
  ::TensorExpressions::evaluate<ti_T, ti_B, ti_t>(make_not_null(&T_contracted),
                                                  T(ti_a, ti_t, ti_A, ti_B));

  for (size_t b = 0; b < 4; b++) {
    DataType expected_T_sum = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t a = 0; a < 4; a++) {
      expected_T_sum += T.get(a, 0, a, b);
    }
    CHECK_ITERABLE_APPROX(T_contracted.get(0, b, 0), expected_T_sum);

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        CHECK(T_contracted.get(i + 1, b, j + 1) ==
              spatial_component_placeholder);
      }
    }
  }
}

template <typename DataType>
void test_contractions(const DataType& used_for_size) {
  test_contractions_rank2(used_for_size);
  test_contractions_rank3(used_for_size);
  test_contractions_rank4(used_for_size);
  test_time_index(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  test_contractions(std::numeric_limits<double>::signaling_NaN());
  test_contractions(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
