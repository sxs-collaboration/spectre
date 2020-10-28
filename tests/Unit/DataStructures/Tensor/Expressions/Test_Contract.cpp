// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"

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
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rul(used_for_size);
  create_tensor(make_not_null(&Rul));

  const Tensor<DataType> RIi_contracted =
      TensorExpressions::evaluate(Rul(ti_I, ti_i));

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

  const Tensor<DataType> RgG_contracted =
      TensorExpressions::evaluate(Rlu(ti_g, ti_G));

  DataType expected_RgG_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t g = 0; g < 4; g++) {
    expected_RgG_sum += Rlu.get(g, g);
  }
  CHECK(RgG_contracted.get() == expected_RgG_sum);
}

template <typename DataType>
void test_contractions_rank3(const DataType& used_for_size) noexcept {
  // Contract first and second indices of (lower, upper, lower) tensor
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  Tensor<DataType, Symmetry<3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<3, UpLo::Up, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Rlul(used_for_size);
  create_tensor(make_not_null(&Rlul));

  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      RiIj_contracted =
          TensorExpressions::evaluate<ti_j_t>(Rlul(ti_i, ti_I, ti_j));

  for (size_t j = 0; j < 4; j++) {
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

  const Tensor<DataType, Symmetry<1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RJLj_contracted =
          TensorExpressions::evaluate<ti_L_t>(Ruul(ti_J, ti_L, ti_j));

  for (size_t l = 0; l < 3; l++) {
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

  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      RBfF_contracted =
          TensorExpressions::evaluate<ti_B_t>(Rulu(ti_B, ti_f, ti_F));

  for (size_t b = 0; b < 4; b++) {
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

  const Tensor<DataType, Symmetry<1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RiaI_contracted =
          TensorExpressions::evaluate<ti_a_t>(Rllu(ti_i, ti_a, ti_I));

  for (size_t a = 0; a < 4; a++) {
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
  // Use explicit type (vs auto) so the compiler checks the return type of
  // `evaluate`
  Tensor<DataType, Symmetry<4, 3, 2, 1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                    SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Rluul(used_for_size);
  create_tensor(make_not_null(&Rluul));

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      RiIKj_contracted = TensorExpressions::evaluate<ti_K_t, ti_j_t>(
          Rluul(ti_i, ti_I, ti_K, ti_j));

  for (size_t k = 0; k < 4; k++) {
    for (size_t j = 0; j < 3; j++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      RABac_contracted = TensorExpressions::evaluate<ti_B_t, ti_c_t>(
          Ruull(ti_A, ti_B, ti_a, ti_c));

  for (size_t b = 0; b < 4; b++) {
    for (size_t c = 0; c < 5; c++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RLJIl_contracted = TensorExpressions::evaluate<ti_J_t, ti_I_t>(
          Ruuul(ti_L, ti_J, ti_I, ti_l));

  for (size_t j = 0; j < 4; j++) {
    for (size_t i = 0; i < 3; i++) {
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

  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      REDdA_contracted = TensorExpressions::evaluate<ti_E_t, ti_A_t>(
          Ruulu(ti_E, ti_D, ti_d, ti_A));

  for (size_t e = 0; e < 4; e++) {
    for (size_t a = 0; a < 4; a++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      RkJij_contracted = TensorExpressions::evaluate<ti_k_t, ti_i_t>(
          Rlull(ti_k, ti_J, ti_i, ti_j));

  for (size_t k = 0; k < 3; k++) {
    for (size_t i = 0; i < 4; i++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<4, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      RFcgG_contracted = TensorExpressions::evaluate<ti_F_t, ti_c_t>(
          Rullu(ti_F, ti_c, ti_g, ti_G));

  for (size_t f = 0; f < 5; f++) {
    for (size_t c = 0; c < 4; c++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<2, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      RKkIJ_contracted_to_JI = TensorExpressions::evaluate<ti_J_t, ti_I_t>(
          Ruluu(ti_K, ti_k, ti_I, ti_J));

  for (size_t j = 0; j < 2; j++) {
    for (size_t i = 0; i < 3; i++) {
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

  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<2, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<2, UpLo::Up, Frame::Grid>>>
      RbCBE_contracted_to_EC = TensorExpressions::evaluate<ti_E_t, ti_C_t>(
          Rluuu(ti_b, ti_C, ti_B, ti_E));

  for (size_t e = 0; e < 3; e++) {
    for (size_t c = 0; c < 3; c++) {
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

  const Tensor<DataType, Symmetry<1, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      RAdba_contracted_to_bd = TensorExpressions::evaluate<ti_b_t, ti_d_t>(
          Rulll(ti_A, ti_d, ti_b, ti_a));

  for (size_t b = 0; b < 4; b++) {
    for (size_t d = 0; d < 4; d++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      RljJi_contracted_to_il = TensorExpressions::evaluate<ti_i_t, ti_l_t>(
          Rllul(ti_l, ti_j, ti_J, ti_i));

  for (size_t i = 0; i < 4; i++) {
    for (size_t l = 0; l < 3; l++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      RagDG_contracted_to_Da = TensorExpressions::evaluate<ti_D_t, ti_a_t>(
          Rlluu(ti_a, ti_g, ti_D, ti_G));

  for (size_t d = 0; d < 4; d++) {
    for (size_t a = 0; a < 4; a++) {
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

  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>,
                          SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      RlJiI_contracted_to_Jl = TensorExpressions::evaluate<ti_J_t, ti_l_t>(
          Rlulu(ti_l, ti_J, ti_i, ti_I));

  for (size_t j = 0; j < 3; j++) {
    for (size_t l = 0; l < 3; l++) {
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

  const Tensor<DataType> RKkLl_contracted =
      TensorExpressions::evaluate(Rulul(ti_K, ti_k, ti_L, ti_l));

  DataType expected_RKkLl_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t k = 0; k < 3; k++) {
    for (size_t l = 0; l < 4; l++) {
      expected_RKkLl_sum += Rulul.get(k, k, l, l);
    }
  }
  CHECK(RKkLl_contracted.get() == expected_RKkLl_sum);

  // Contract first and third indices and second and fourth indices to rank 0
  // tensor
  const Tensor<DataType> RcaCA_contracted =
      TensorExpressions::evaluate(Rlluu(ti_c, ti_a, ti_C, ti_A));

  DataType expected_RcaCA_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t c = 0; c < 4; c++) {
    for (size_t a = 0; a < 4; a++) {
      expected_RcaCA_sum += Rlluu.get(c, a, c, a);
    }
  }
  CHECK(RcaCA_contracted.get() == expected_RcaCA_sum);

  // Contract first and fourth indices and second and third indices to rank 0
  // tensor
  const Tensor<DataType> RjIiJ_contracted =
      TensorExpressions::evaluate(Rlulu(ti_j, ti_I, ti_i, ti_J));

  DataType expected_RjIiJ_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t j = 0; j < 3; j++) {
    for (size_t i = 0; i < 3; i++) {
      expected_RjIiJ_sum += Rlulu.get(j, i, i, j);
    }
  }
  CHECK(RjIiJ_contracted.get() == expected_RjIiJ_sum);
}

template <typename DataType>
void test_contractions(const DataType& used_for_size) noexcept {
  test_contractions_rank2(used_for_size);
  test_contractions_rank3(used_for_size);
  test_contractions_rank4(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Contract",
                  "[DataStructures][Unit]") {
  test_contractions(std::numeric_limits<double>::signaling_NaN());
  test_contractions(
      DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
