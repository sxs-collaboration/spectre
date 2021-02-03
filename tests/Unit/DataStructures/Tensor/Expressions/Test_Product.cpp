// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iterator>
#include <numeric>

#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/Product.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<double, Ts...>*> tensor) noexcept {
  std::iota(tensor->begin(), tensor->end(), 0.0);
}

template <typename... Ts>
void assign_unique_values_to_tensor(
    gsl::not_null<Tensor<DataVector, Ts...>*> tensor) noexcept {
  double value = 0.0;
  for (auto index_it = tensor->begin(); index_it != tensor->end(); index_it++) {
    for (auto vector_it = index_it->begin(); vector_it != index_it->end();
         vector_it++) {
      *vector_it = value;
      value += 1.0;
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of a `double` and tensor is correctly
/// evaluated
///
/// \details
/// The outer product cases tested are:
/// - \f$L_{ij} = R * S_{ij}\f$
/// - \f$L_{ij} = S_{ij} * R\f$
/// - \f$L_{ij} = R * S_{ij} * T\f$
///
/// where \f$R\f$ and \f$T\f$ are `double`s and \f$S\f$ and \f$L\f$ are Tensors
/// with data type `double` or DataVector.
///
/// \tparam DataType the type of data being stored in the tensor operand of the
/// products
template <typename DataType>
void test_outer_product_double(const DataType& used_for_size) noexcept {
  constexpr size_t dim = 3;
  using tensor_type =
      Tensor<DataType, Symmetry<1, 1>,
             index_list<SpatialIndex<dim, UpLo::Lo, Frame::Inertial>,
                        SpatialIndex<dim, UpLo::Lo, Frame::Inertial>>>;

  tensor_type S(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&S));

  // \f$L_{ij} = R * S_{ij}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const tensor_type Lij_from_R_Sij =
      TensorExpressions::evaluate<ti_i, ti_j>(5.6 * S(ti_i, ti_j));
  // \f$L_{ij} = S_{ij} * R\f$
  const tensor_type Lij_from_Sij_R =
      TensorExpressions::evaluate<ti_i, ti_j>(S(ti_i, ti_j) * -8.1);
  // \f$L_{ij} = R * S_{ij} * T\f$
  const tensor_type Lij_from_R_Sij_T =
      TensorExpressions::evaluate<ti_i, ti_j>(-1.7 * S(ti_i, ti_j) * 0.6);

  for (size_t i = 0; i < dim; i++) {
    for (size_t j = 0; j < dim; j++) {
      CHECK(Lij_from_R_Sij.get(i, j) == 5.6 * S.get(i, j));
      CHECK(Lij_from_Sij_R.get(i, j) == S.get(i, j) * -8.1);
      CHECK(Lij_from_R_Sij_T.get(i, j) == -1.7 * S.get(i, j) * 0.6);
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of a rank 0 tensor with another tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are:
/// - \f$L = R * R\f$
/// - \f$L = R * R * R\f$
/// - \f$L^{a} = R * S^{a}\f$
/// - \f$L_{ai} = R * T_{ai}\f$
///
/// For the last two cases, both operand orderings are tested. For the last
/// case, both LHS index orderings are tested.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_outer_product_rank_0_operand(const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` to a proper test value
    R.get() = -3.7;
  } else {
    assign_unique_values_to_tensor(make_not_null(&R));
  }

  // \f$L = R * R\f$
  CHECK(TensorExpressions::evaluate(R() * R()).get() == R.get() * R.get());
  // \f$L = R * R * R\f$
  CHECK(TensorExpressions::evaluate(R() * R() * R()).get() ==
        R.get() * R.get() * R.get());

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Inertial>>>
      Su(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Su));

  // \f$L^{a} = R * S^{a}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const decltype(Su) LA_from_R_SA =
      TensorExpressions::evaluate<ti_A>(R() * Su(ti_A));
  // \f$L^{a} = S^{a} * R\f$
  const decltype(Su) LA_from_SA_R =
      TensorExpressions::evaluate<ti_A>(Su(ti_A) * R());

  for (size_t a = 0; a < 4; a++) {
    CHECK(LA_from_R_SA.get(a) == R.get() * Su.get(a));
    CHECK(LA_from_SA_R.get(a) == Su.get(a) * R.get());
  }

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                    SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Tll(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Tll));

  // \f$L_{ai} = R * T_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Lai_from_R_Tai =
          TensorExpressions::evaluate<ti_a, ti_i>(R() * Tll(ti_a, ti_i));
  // \f$L_{ia} = R * T_{ai}\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Lia_from_R_Tai =
          TensorExpressions::evaluate<ti_i, ti_a>(R() * Tll(ti_a, ti_i));
  // \f$L_{ai} = T_{ai} * R\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>,
                          SpatialIndex<4, UpLo::Lo, Frame::Inertial>>>
      Lai_from_Tai_R =
          TensorExpressions::evaluate<ti_a, ti_i>(Tll(ti_a, ti_i) * R());
  // \f$L_{ia} = T_{ai} * R\f$
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Inertial>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>>>
      Lia_from_Tai_R =
          TensorExpressions::evaluate<ti_i, ti_a>(Tll(ti_a, ti_i) * R());

  for (size_t a = 0; a < 4; a++) {
    for (size_t i = 0; i < 4; i++) {
      const DataType expected_R_Tai_product = R.get() * Tll.get(a, i);
      CHECK(Lai_from_R_Tai.get(a, i) == expected_R_Tai_product);
      CHECK(Lia_from_R_Tai.get(i, a) == expected_R_Tai_product);

      const DataType expected_Tai_R_product = Tll.get(a, i) * R.get();
      CHECK(Lai_from_Tai_R.get(a, i) == expected_Tai_R_product);
      CHECK(Lia_from_Tai_R.get(i, a) == expected_Tai_R_product);
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of rank 1 tensors with another tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are:
/// - \f$L^{a}{}_{i} = R_{i} * S^{a}\f$
/// - \f$L^{ja}{}_{i} = R_{i} * S^{a} * T^{j}\f$
/// - \f$L_{k}{}^{c}{}_{d} = S^{c} * G_{dk}\f$
/// - \f$L^{c}{}_{dk} = G_{dk} * S^{c}\f$
///
/// For each case, the LHS index order is different from the order in the
/// operands.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_outer_product_rank_1_operand(const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      Rl(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rl));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Su(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Su));

  // \f$L^{a}{}_{i} = R_{i} * S^{a}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      LAi_from_Ri_SA =
          TensorExpressions::evaluate<ti_A, ti_i>(Rl(ti_i) * Su(ti_A));

  for (size_t i = 0; i < 3; i++) {
    for (size_t a = 0; a < 4; a++) {
      CHECK(LAi_from_Ri_SA.get(a, i) == Rl.get(i) * Su.get(a));
    }
  }

  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>>>
      Tu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Tu));

  // \f$L^{ja}{}_{i} = R_{i} * S^{a} * T^{j}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      LJAi_from_Ri_SA_TJ = TensorExpressions::evaluate<ti_J, ti_A, ti_i>(
          Rl(ti_i) * Su(ti_A) * Tu(ti_J));

  for (size_t j = 0; j < 3; j++) {
    for (size_t a = 0; a < 4; a++) {
      for (size_t i = 0; i < 3; i++) {
        CHECK(LJAi_from_Ri_SA_TJ.get(j, a, i) ==
              Rl.get(i) * Su.get(a) * Tu.get(j));
      }
    }
  }

  Tensor<DataType, Symmetry<2, 1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                    SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      Gll(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Gll));

  // \f$L_{k}{}^{c}{}_{d} = S^{c} * G_{dk}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpatialIndex<4, UpLo::Lo, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      LkCd_from_SC_Gdk = TensorExpressions::evaluate<ti_k, ti_C, ti_d>(
          Su(ti_C) * Gll(ti_d, ti_k));
  // \f$L^{c}{}_{dk} = G_{dk} * S^{c}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>,
                          SpacetimeIndex<3, UpLo::Lo, Frame::Grid>,
                          SpatialIndex<4, UpLo::Lo, Frame::Grid>>>
      LCdk_from_Gdk_SC = TensorExpressions::evaluate<ti_C, ti_d, ti_k>(
          Gll(ti_d, ti_k) * Su(ti_C));

  for (size_t k = 0; k < 4; k++) {
    for (size_t c = 0; c < 4; c++) {
      for (size_t d = 0; d < 4; d++) {
        CHECK(LkCd_from_SC_Gdk.get(k, c, d) == Su.get(c) * Gll.get(d, k));
        CHECK(LCdk_from_Gdk_SC.get(c, d, k) == Gll.get(d, k) * Su.get(c));
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of two rank 2 tensors is correctly evaluated
///
/// \details
/// All LHS index orders are tested. One such example case:
/// \f$L_{abc}{}^{i} = R_{ab} * S^{i}{}_{c}\f$
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_outer_product_rank_2x2_operands(
    const DataType& used_for_size) noexcept {
  using R_index = SpacetimeIndex<3, UpLo::Lo, Frame::Grid>;
  using S_first_index = SpatialIndex<4, UpLo::Up, Frame::Grid>;
  using S_second_index = SpacetimeIndex<2, UpLo::Lo, Frame::Grid>;

  Tensor<DataType, Symmetry<1, 1>, index_list<R_index, R_index>> Rll(
      used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rll));
  Tensor<DataType, Symmetry<2, 1>, index_list<S_first_index, S_second_index>>
      Sul(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sul));

  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_first_index, S_second_index>>
      L_abIc = TensorExpressions::evaluate<ti_a, ti_b, ti_I, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_second_index, S_first_index>>
      L_abcI = TensorExpressions::evaluate<ti_a, ti_b, ti_c, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_first_index, R_index, S_second_index>>
      L_aIbc = TensorExpressions::evaluate<ti_a, ti_I, ti_b, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_first_index, S_second_index, R_index>>
      L_aIcb = TensorExpressions::evaluate<ti_a, ti_I, ti_c, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_second_index, R_index, S_first_index>>
      L_acbI = TensorExpressions::evaluate<ti_a, ti_c, ti_b, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_second_index, S_first_index, R_index>>
      L_acIb = TensorExpressions::evaluate<ti_a, ti_c, ti_I, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_first_index, S_second_index>>
      L_baIc = TensorExpressions::evaluate<ti_b, ti_a, ti_I, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 3, 2, 1>,
               index_list<R_index, R_index, S_second_index, S_first_index>>
      L_bacI = TensorExpressions::evaluate<ti_b, ti_a, ti_c, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_first_index, R_index, S_second_index>>
      L_bIac = TensorExpressions::evaluate<ti_b, ti_I, ti_a, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_first_index, S_second_index, R_index>>
      L_bIca = TensorExpressions::evaluate<ti_b, ti_I, ti_c, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 3, 1>,
               index_list<R_index, S_second_index, R_index, S_first_index>>
      L_bcaI = TensorExpressions::evaluate<ti_b, ti_c, ti_a, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 3>,
               index_list<R_index, S_second_index, S_first_index, R_index>>
      L_bcIa = TensorExpressions::evaluate<ti_b, ti_c, ti_I, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_first_index, R_index, R_index, S_second_index>>
      L_Iabc = TensorExpressions::evaluate<ti_I, ti_a, ti_b, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_first_index, R_index, S_second_index, R_index>>
      L_Iacb = TensorExpressions::evaluate<ti_I, ti_a, ti_c, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_first_index, R_index, R_index, S_second_index>>
      L_Ibac = TensorExpressions::evaluate<ti_I, ti_b, ti_a, ti_c>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_first_index, R_index, S_second_index, R_index>>
      L_Ibca = TensorExpressions::evaluate<ti_I, ti_b, ti_c, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_first_index, S_second_index, R_index, R_index>>
      L_Icab = TensorExpressions::evaluate<ti_I, ti_c, ti_a, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_first_index, S_second_index, R_index, R_index>>
      L_Icba = TensorExpressions::evaluate<ti_I, ti_c, ti_b, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_second_index, R_index, R_index, S_first_index>>
      L_cabI = TensorExpressions::evaluate<ti_c, ti_a, ti_b, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_second_index, R_index, S_first_index, R_index>>
      L_caIb = TensorExpressions::evaluate<ti_c, ti_a, ti_I, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 2, 1>,
               index_list<S_second_index, R_index, R_index, S_first_index>>
      L_cbaI = TensorExpressions::evaluate<ti_c, ti_b, ti_a, ti_I>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 2>,
               index_list<S_second_index, R_index, S_first_index, R_index>>
      L_cbIa = TensorExpressions::evaluate<ti_c, ti_b, ti_I, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_second_index, S_first_index, R_index, R_index>>
      L_cIab = TensorExpressions::evaluate<ti_c, ti_I, ti_a, ti_b>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));
  const Tensor<DataType, Symmetry<3, 2, 1, 1>,
               index_list<S_second_index, S_first_index, R_index, R_index>>
      L_cIba = TensorExpressions::evaluate<ti_c, ti_I, ti_b, ti_a>(
          Rll(ti_a, ti_b) * Sul(ti_I, ti_c));

  for (size_t a = 0; a < R_index::dim; a++) {
    for (size_t b = 0; b < R_index::dim; b++) {
      for (size_t i = 0; i < S_first_index::dim; i++) {
        for (size_t c = 0; c < S_second_index::dim; c++) {
          CHECK(L_abIc.get(a, b, i, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_abcI.get(a, b, c, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_aIbc.get(a, i, b, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_aIcb.get(a, i, c, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_acbI.get(a, c, b, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_acIb.get(a, c, i, b) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_baIc.get(b, a, i, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bacI.get(b, a, c, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bIac.get(b, i, a, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bIca.get(b, i, c, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bcaI.get(b, c, a, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_bcIa.get(b, c, i, a) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_Iabc.get(i, a, b, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Iacb.get(i, a, c, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Ibac.get(i, b, a, c) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Ibca.get(i, b, c, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Icab.get(i, c, a, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_Icba.get(i, c, b, a) == Rll.get(a, b) * Sul.get(i, c));

          CHECK(L_cabI.get(c, a, b, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_caIb.get(c, a, i, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cbaI.get(c, b, a, i) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cbIa.get(c, b, i, a) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cIab.get(c, i, a, b) == Rll.get(a, b) * Sul.get(i, c));
          CHECK(L_cIba.get(c, i, b, a) == Rll.get(a, b) * Sul.get(i, c));
        }
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the outer product of a rank 0, rank 1, and rank 2 tensor is
/// correctly evaluated
///
/// \details
/// The outer product cases tested are permutations of the form:
/// - \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
///
/// Each case represents an ordering for the operands and the LHS indices.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_outer_product_rank_0x1x2_operands(
    const DataType& used_for_size) noexcept {
  Tensor<DataType> R{{{used_for_size}}};
  if constexpr (std::is_same_v<DataType, double>) {
    // Replace tensor's value from `used_for_size` to a proper test value
    R.get() = 4.5;
  } else {
    assign_unique_values_to_tensor(make_not_null(&R));
  }

  using S_index = SpacetimeIndex<3, UpLo::Up, Frame::Inertial>;
  using T_first_index = SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>;
  using T_second_index = SpatialIndex<4, UpLo::Lo, Frame::Inertial>;

  Tensor<DataType, Symmetry<1>, index_list<S_index>> Su(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Su));

  Tensor<DataType, Symmetry<2, 1>, index_list<T_first_index, T_second_index>>
      Tll(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Tll));

  // \f$R * S^{a} * T_{bi}\f$
  const auto R_SA_Tbi_expr = R() * Su(ti_A) * Tll(ti_b, ti_i);
  // \f$L^{a}{}_{bi} = R * S^{a} * T_{bi}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(R_SA_Tbi_expr);
  // \f$L^{a}{}_{ib} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(R_SA_Tbi_expr);
  // \f$L_{b}{}^{a}{}_{i} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(R_SA_Tbi_expr);
  // \f$L_{bi}{}^{a} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(R_SA_Tbi_expr);
  // \f$L_{i}{}^{a}{}_{b} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(R_SA_Tbi_expr);
  // \f$L_{ib}{}^{a} = R * S^{a} * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_R_SA_Tbi =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(R_SA_Tbi_expr);

  // \f$R * T_{bi} * S^{a}\f$
  const auto R_Tbi_SA_expr = R() * Tll(ti_b, ti_i) * Su(ti_A);
  // \f$L^{a}{}_{bi} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(R_Tbi_SA_expr);
  // \f$L^{a}{}_{ib} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(R_Tbi_SA_expr);
  // \f$L_{b}{}^{a}{}_{i} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(R_Tbi_SA_expr);
  // \f$L_{bi}{}^{a} = R * R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(R_Tbi_SA_expr);
  // \f$L_{i}{}^{a}{}_{b} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(R_Tbi_SA_expr);
  // \f$L_{ib}{}^{a} = R * T_{bi} * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_R_Tbi_SA =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(R_Tbi_SA_expr);

  // \f$S^{a} * R * T_{bi}\f$
  const auto SA_R_Tbi_expr = Su(ti_A) * R() * Tll(ti_b, ti_i);
  // \f$L^{a}{}_{bi} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(SA_R_Tbi_expr);
  // \f$L^{a}{}_{ib} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(SA_R_Tbi_expr);
  // \f$L_{b}{}^{a}{}_{i} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(SA_R_Tbi_expr);
  // \f$L_{bi}{}^{a} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(SA_R_Tbi_expr);
  // \f$L_{i}{}^{a}{}_{b} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(SA_R_Tbi_expr);
  // \f$L_{ib}{}^{a} = S^{a} * R * T_{bi}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_SA_R_Tbi =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(SA_R_Tbi_expr);

  // \f$S^{a} * T_{bi} * R\f$
  const auto SA_Tbi_R_expr = Su(ti_A) * Tll(ti_b, ti_i) * R();
  // \f$L^{a}{}_{bi} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(SA_Tbi_R_expr);
  // \f$L^{a}{}_{ib} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(SA_Tbi_R_expr);
  // \f$L_{b}{}^{a}{}_{i} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(SA_Tbi_R_expr);
  // \f$L_{bi}{}^{a} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(SA_Tbi_R_expr);
  // \f$L_{i}{}^{a}{}_{b} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(SA_Tbi_R_expr);
  // \f$L_{ib}{}^{a} = S^{a} * T_{bi} * R\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_SA_Tbi_R =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(SA_Tbi_R_expr);

  // \f$T_{bi} * R * S^{a}\f$
  const auto Tbi_R_SA_expr = Tll(ti_b, ti_i) * R() * Su(ti_A);
  // \f$L^{a}{}_{bi} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(Tbi_R_SA_expr);
  // \f$L^{a}{}_{ib} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(Tbi_R_SA_expr);
  // \f$L_{b}{}^{a}{}_{i} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(Tbi_R_SA_expr);
  // \f$L_{bi}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(Tbi_R_SA_expr);
  // \f$L_{i}{}^{a}{}_{b} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(Tbi_R_SA_expr);
  // \f$L_{ib}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_Tbi_R_SA =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(Tbi_R_SA_expr);

  // \f$T_{bi} * S^{a} * R\f$
  const auto Tbi_SA_R_expr = Tll(ti_b, ti_i) * Su(ti_A) * R();
  // \f$L^{a}{}_{bi} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_first_index, T_second_index>>
      LAbi_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_A, ti_b, ti_i>(Tbi_SA_R_expr);
  // \f$L^{a}{}_{ib} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<S_index, T_second_index, T_first_index>>
      LAib_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_A, ti_i, ti_b>(Tbi_SA_R_expr);
  // \f$L_{b}{}^{a}{}_{i} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, S_index, T_second_index>>
      LbAi_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_b, ti_A, ti_i>(Tbi_SA_R_expr);
  // \f$L_{bi}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_first_index, T_second_index, S_index>>
      LbiA_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_b, ti_i, ti_A>(Tbi_SA_R_expr);
  // \f$L_{i}{}^{a}{}_{b} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, S_index, T_first_index>>
      LiAb_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_i, ti_A, ti_b>(Tbi_SA_R_expr);
  // \f$L_{ib}{}^{a} = T_{bi} * R * S^{a}\f$
  const Tensor<DataType, Symmetry<3, 2, 1>,
               index_list<T_second_index, T_first_index, S_index>>
      LibA_from_Tbi_SA_R =
          TensorExpressions::evaluate<ti_i, ti_b, ti_A>(Tbi_SA_R_expr);

  for (size_t a = 0; a < S_index::dim; a++) {
    for (size_t b = 0; b < T_first_index::dim; b++) {
      for (size_t i = 0; i < T_second_index::dim; i++) {
        const DataType expected_product = R.get() * Su.get(a) * Tll.get(b, i);

        CHECK(LAbi_from_R_SA_Tbi.get(a, b, i) == expected_product);
        CHECK(LAib_from_R_SA_Tbi.get(a, i, b) == expected_product);
        CHECK(LbAi_from_R_SA_Tbi.get(b, a, i) == expected_product);
        CHECK(LbiA_from_R_SA_Tbi.get(b, i, a) == expected_product);
        CHECK(LiAb_from_R_SA_Tbi.get(i, a, b) == expected_product);
        CHECK(LibA_from_R_SA_Tbi.get(i, b, a) == expected_product);

        CHECK(LAbi_from_R_Tbi_SA.get(a, b, i) == expected_product);
        CHECK(LAib_from_R_Tbi_SA.get(a, i, b) == expected_product);
        CHECK(LbAi_from_R_Tbi_SA.get(b, a, i) == expected_product);
        CHECK(LbiA_from_R_Tbi_SA.get(b, i, a) == expected_product);
        CHECK(LiAb_from_R_Tbi_SA.get(i, a, b) == expected_product);
        CHECK(LibA_from_R_Tbi_SA.get(i, b, a) == expected_product);

        CHECK(LAbi_from_SA_R_Tbi.get(a, b, i) == expected_product);
        CHECK(LAib_from_SA_R_Tbi.get(a, i, b) == expected_product);
        CHECK(LbAi_from_SA_R_Tbi.get(b, a, i) == expected_product);
        CHECK(LbiA_from_SA_R_Tbi.get(b, i, a) == expected_product);
        CHECK(LiAb_from_SA_R_Tbi.get(i, a, b) == expected_product);
        CHECK(LibA_from_SA_R_Tbi.get(i, b, a) == expected_product);

        CHECK(LAbi_from_SA_Tbi_R.get(a, b, i) == expected_product);
        CHECK(LAib_from_SA_Tbi_R.get(a, i, b) == expected_product);
        CHECK(LbAi_from_SA_Tbi_R.get(b, a, i) == expected_product);
        CHECK(LbiA_from_SA_Tbi_R.get(b, i, a) == expected_product);
        CHECK(LiAb_from_SA_Tbi_R.get(i, a, b) == expected_product);
        CHECK(LibA_from_SA_Tbi_R.get(i, b, a) == expected_product);

        CHECK(LAbi_from_Tbi_R_SA.get(a, b, i) == expected_product);
        CHECK(LAib_from_Tbi_R_SA.get(a, i, b) == expected_product);
        CHECK(LbAi_from_Tbi_R_SA.get(b, a, i) == expected_product);
        CHECK(LbiA_from_Tbi_R_SA.get(b, i, a) == expected_product);
        CHECK(LiAb_from_Tbi_R_SA.get(i, a, b) == expected_product);
        CHECK(LibA_from_Tbi_R_SA.get(i, b, a) == expected_product);

        CHECK(LAbi_from_Tbi_SA_R.get(a, b, i) == expected_product);
        CHECK(LAib_from_Tbi_SA_R.get(a, i, b) == expected_product);
        CHECK(LbAi_from_Tbi_SA_R.get(b, a, i) == expected_product);
        CHECK(LbiA_from_Tbi_SA_R.get(b, i, a) == expected_product);
        CHECK(LiAb_from_Tbi_SA_R.get(i, a, b) == expected_product);
        CHECK(LibA_from_Tbi_SA_R.get(i, b, a) == expected_product);
      }
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the inner product of two rank 1 tensors is correctly evaluated
///
/// \details
/// The inner product cases tested are:
/// - \f$L = R^{a} * S_{a}\f$
/// - \f$L = S_{a} * R^{a}\f$
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_inner_product_rank_1x1_operands(
    const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Up, Frame::Grid>>>
      Ru(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Ru));

  Tensor<DataType, Symmetry<1>,
         index_list<SpacetimeIndex<3, UpLo::Lo, Frame::Grid>>>
      Sl(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sl));

  // \f$L = R^{a} * S_{a}\f$
  const Tensor<DataType> L_from_RA_Sa =
      TensorExpressions::evaluate(Ru(ti_A) * Sl(ti_a));
  // \f$L = S_{a} * R^{a}\f$
  const Tensor<DataType> L_from_Sa_RA =
      TensorExpressions::evaluate(Sl(ti_a) * Ru(ti_A));

  DataType expected_sum = make_with_value<DataType>(used_for_size, 0.0);
  for (size_t a = 0; a < 4; a++) {
    expected_sum += (Ru.get(a) * Sl.get(a));
  }
  CHECK(L_from_RA_Sa.get() == expected_sum);
  CHECK(L_from_Sa_RA.get() == expected_sum);
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the inner product of two rank 2 tensors is correctly evaluated
///
/// \details
/// All cases in this test contract both pairs of indices of the two rank 2
/// tensor operands to a resulting rank 0 tensor. For each case, the two tensor
/// operands have one spacetime and one spatial index. Each case is a
/// permutation of the positions of contracted pairs and their valences. One
/// such example case: \f$L = R_{ai} * S^{ai}\f$
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_inner_product_rank_2x2_operands(
    const DataType& used_for_size) noexcept {
  using lower_spacetime_index = SpacetimeIndex<3, UpLo::Lo, Frame::Inertial>;
  using upper_spacetime_index = SpacetimeIndex<3, UpLo::Up, Frame::Inertial>;
  using lower_spatial_index = SpatialIndex<2, UpLo::Lo, Frame::Inertial>;
  using upper_spatial_index = SpatialIndex<2, UpLo::Up, Frame::Inertial>;

  // All tensor variables starting with 'R' refer to tensors whose first index
  // is a spacetime index and whose second index is a spatial index. Conversely,
  // all tensor variables starting with 'S' refer to tensors whose first index
  // is a spatial index and whose second index is a spacetime index.
  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spacetime_index, lower_spatial_index>>
      Rll(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rll));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spatial_index, lower_spacetime_index>>
      Sll(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sll));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spacetime_index, upper_spatial_index>>
      Ruu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Ruu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spatial_index, upper_spacetime_index>>
      Suu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Suu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spacetime_index, upper_spatial_index>>
      Rlu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rlu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<lower_spatial_index, upper_spacetime_index>>
      Slu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Slu));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spacetime_index, lower_spatial_index>>
      Rul(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rul));
  Tensor<DataType, Symmetry<2, 1>,
         index_list<upper_spatial_index, lower_spacetime_index>>
      Sul(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sul));

  // \f$L = Rll_{ai} * Ruu^{ai}\f$
  const Tensor<DataType> L_aiAI_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_i) * Ruu(ti_A, ti_I));
  // \f$L = Rll_{ai} * Suu^{ia}\f$
  const Tensor<DataType> L_aiIA_product =
      TensorExpressions::evaluate(Rll(ti_a, ti_i) * Suu(ti_I, ti_A));
  // \f$L = Ruu^{ai} * Rll_{ai}\f$
  const Tensor<DataType> L_AIai_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_I) * Rll(ti_a, ti_i));
  // \f$L = Ruu^{ai} * Sll_{ia}\f$
  const Tensor<DataType> L_AIia_product =
      TensorExpressions::evaluate(Ruu(ti_A, ti_I) * Sll(ti_i, ti_a));
  // \f$L = Rlu_{a}{}^{i} * Rul^{a}{}_{i}\f$
  const Tensor<DataType> L_aIAi_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_I) * Rul(ti_A, ti_i));
  // \f$L = Rlu_{a}{}^{i} * Slu_{i}{}^{a}\f$
  const Tensor<DataType> L_aIiA_product =
      TensorExpressions::evaluate(Rlu(ti_a, ti_I) * Slu(ti_i, ti_A));
  // \f$L = Rul^{a}{}_{i} * Rlu_{a}{}^{i}\f$
  const Tensor<DataType> L_AiaI_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_i) * Rlu(ti_a, ti_I));
  // \f$L = Rul^{a}{}_{i} * Sul^{i}{}_{a}\f$
  const Tensor<DataType> L_AiIa_product =
      TensorExpressions::evaluate(Rul(ti_A, ti_i) * Sul(ti_I, ti_a));

  DataType L_aiAI_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_aiIA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AIai_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AIia_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_aIAi_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_aIiA_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AiaI_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);
  DataType L_AiIa_expected_product =
      make_with_value<DataType>(used_for_size, 0.0);

  for (size_t a = 0; a < lower_spacetime_index::dim; a++) {
    for (size_t i = 0; i < lower_spatial_index::dim; i++) {
      L_aiAI_expected_product += (Rll.get(a, i) * Ruu.get(a, i));
      L_aiIA_expected_product += (Rll.get(a, i) * Suu.get(i, a));
      L_AIai_expected_product += (Ruu.get(a, i) * Rll.get(a, i));
      L_AIia_expected_product += (Ruu.get(a, i) * Sll.get(i, a));
      L_aIAi_expected_product += (Rlu.get(a, i) * Rul.get(a, i));
      L_aIiA_expected_product += (Rlu.get(a, i) * Slu.get(i, a));
      L_AiaI_expected_product += (Rul.get(a, i) * Rlu.get(a, i));
      L_AiIa_expected_product += (Rul.get(a, i) * Sul.get(i, a));
    }
  }
  CHECK(L_aiAI_product.get() == L_aiAI_expected_product);
  CHECK(L_aiIA_product.get() == L_aiIA_expected_product);
  CHECK(L_AIai_product.get() == L_AIai_expected_product);
  CHECK(L_AIia_product.get() == L_AIia_expected_product);
  CHECK(L_aIAi_product.get() == L_aIAi_expected_product);
  CHECK(L_aIiA_product.get() == L_aIiA_expected_product);
  CHECK(L_AiaI_product.get() == L_AiaI_expected_product);
  CHECK(L_AiIa_product.get() == L_AiIa_expected_product);
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the product of two tensors with one pair of indices to contract
/// is correctly evaluated
///
/// \details
/// The product cases tested are:
/// - \f$L_{b} = R_{ab} * T^{a}\f$
/// - \f$L_{ac} = R_{ab} * S^{b}_{c}\f$
///
/// All cases in this test contract one pair of indices of the two tensor
/// operands. Each case is a permutation of the position of the contracted pair
/// and the ordering of the LHS indices.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_two_term_inner_outer_product(const DataType& used_for_size) noexcept {
  using R_index = SpacetimeIndex<3, UpLo::Lo, Frame::Grid>;
  using T_index = SpacetimeIndex<3, UpLo::Up, Frame::Grid>;

  Tensor<DataType, Symmetry<1, 1>, index_list<R_index, R_index>> Rll(
      used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Rll));
  Tensor<DataType, Symmetry<1>, index_list<T_index>> Tu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Tu));

  // \f$L_{b} = R_{ab} * T^{a}\f$
  // Use explicit type (vs auto) for LHS Tensor so the compiler checks the
  // return type of `evaluate`
  using Lb = Tensor<DataType, Symmetry<1>, index_list<R_index>>;
  const Lb Lb_from_Rab_TA =
      TensorExpressions::evaluate<ti_b>(Rll(ti_a, ti_b) * Tu(ti_A));
  // \f$L_{b} = R_{ba} * T^{a}\f$
  const Lb Lb_from_Rba_TA =
      TensorExpressions::evaluate<ti_b>(Rll(ti_b, ti_a) * Tu(ti_A));
  // \f$L_{b} = T^{a} * R_{ab}\f$
  const Lb Lb_from_TA_Rab =
      TensorExpressions::evaluate<ti_b>(Tu(ti_A) * Rll(ti_a, ti_b));
  // \f$L_{b} = T^{a} * R_{ba}\f$
  const Lb Lb_from_TA_Rba =
      TensorExpressions::evaluate<ti_b>(Tu(ti_A) * Rll(ti_b, ti_a));

  for (size_t b = 0; b < R_index::dim; b++) {
    DataType expected_product = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t a = 0; a < T_index::dim; a++) {
      expected_product += (Rll.get(a, b) * Tu.get(a));
    }
    CHECK(Lb_from_Rab_TA.get(b) == expected_product);
    CHECK(Lb_from_Rba_TA.get(b) == expected_product);
    CHECK(Lb_from_TA_Rab.get(b) == expected_product);
    CHECK(Lb_from_TA_Rba.get(b) == expected_product);
  }

  using S_lower_index = SpacetimeIndex<2, UpLo::Lo, Frame::Grid>;
  using S_upper_index = SpacetimeIndex<3, UpLo::Up, Frame::Grid>;

  Tensor<DataType, Symmetry<2, 1>, index_list<S_upper_index, S_lower_index>>
      Sul(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sul));
  Tensor<DataType, Symmetry<2, 1>, index_list<S_lower_index, S_upper_index>>
      Slu(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Slu));

  // \f$L_{ac} = R_{ab} * S^{b}_{c}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<R_index, S_lower_index>>
      L_abBc_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(Rll(ti_a, ti_b) *
                                                             Sul(ti_B, ti_c));
  // \f$L_{ca} = R_{ab} * S^{b}_{c}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<S_lower_index, R_index>>
      L_abBc_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(Rll(ti_a, ti_b) *
                                                             Sul(ti_B, ti_c));
  // \f$L_{ac} = R_{ab} * S_{c}^{b}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<R_index, S_lower_index>>
      L_abcB_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(Rll(ti_a, ti_b) *
                                                             Slu(ti_c, ti_B));
  // \f$L_{ca} = R_{ab} * S_{c}^{b}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<S_lower_index, R_index>>
      L_abcB_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(Rll(ti_a, ti_b) *
                                                             Slu(ti_c, ti_B));
  // \f$L_{ac} = R_{ba} * S^{b}_{c}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<R_index, S_lower_index>>
      L_baBc_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(Rll(ti_b, ti_a) *
                                                             Sul(ti_B, ti_c));
  // \f$L_{ca} = R_{ba} * S^{b}_{c}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<S_lower_index, R_index>>
      L_baBc_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(Rll(ti_b, ti_a) *
                                                             Sul(ti_B, ti_c));
  // \f$L_{ac} = R_{ba} * S_{c}^{b}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<R_index, S_lower_index>>
      L_bacB_to_ac = TensorExpressions::evaluate<ti_a, ti_c>(Rll(ti_b, ti_a) *
                                                             Slu(ti_c, ti_B));
  // \f$L_{ca} = R_{ba} * S_{c}^{b}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<S_lower_index, R_index>>
      L_bacB_to_ca = TensorExpressions::evaluate<ti_c, ti_a>(Rll(ti_b, ti_a) *
                                                             Slu(ti_c, ti_B));

  for (size_t a = 0; a < R_index::dim; a++) {
    for (size_t c = 0; c < S_lower_index::dim; c++) {
      DataType L_abBc_expected_product =
          make_with_value<DataType>(used_for_size, 0.0);
      DataType L_abcB_expected_product =
          make_with_value<DataType>(used_for_size, 0.0);
      DataType L_baBc_expected_product =
          make_with_value<DataType>(used_for_size, 0.0);
      DataType L_bacB_expected_product =
          make_with_value<DataType>(used_for_size, 0.0);
      for (size_t b = 0; b < 4; b++) {
        L_abBc_expected_product += (Rll.get(a, b) * Sul.get(b, c));
        L_abcB_expected_product += (Rll.get(a, b) * Slu.get(c, b));
        L_baBc_expected_product += (Rll.get(b, a) * Sul.get(b, c));
        L_bacB_expected_product += (Rll.get(b, a) * Slu.get(c, b));
      }
      CHECK(L_abBc_to_ac.get(a, c) == L_abBc_expected_product);
      CHECK(L_abBc_to_ca.get(c, a) == L_abBc_expected_product);
      CHECK(L_abcB_to_ac.get(a, c) == L_abcB_expected_product);
      CHECK(L_abcB_to_ca.get(c, a) == L_abcB_expected_product);
      CHECK(L_baBc_to_ac.get(a, c) == L_baBc_expected_product);
      CHECK(L_baBc_to_ca.get(c, a) == L_baBc_expected_product);
      CHECK(L_bacB_to_ac.get(a, c) == L_bacB_expected_product);
      CHECK(L_bacB_to_ca.get(c, a) == L_bacB_expected_product);
    }
  }
}

/// \ingroup TestingFrameworkGroup
/// \brief Test the product of three tensors involving both inner and outer
/// products of indices is correctly evaluated
///
/// \details
/// The product cases tested are:
/// - \f$L_{i} = R^{j} * S_{j} * T_{i}\f$
/// - \f$L_{i}{}^{k} = S_{j} * T_{i} * G^{jk}\f$
///
/// For each case, multiple operand orderings are tested. For the second case,
/// both LHS index orderings are also tested.
///
/// \tparam DataType the type of data being stored in the product operands
template <typename DataType>
void test_three_term_inner_outer_product(
    const DataType& used_for_size) noexcept {
  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Up, Frame::Inertial>>>
      Ru(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Ru));
  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Sl(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Sl));
  Tensor<DataType, Symmetry<1>,
         index_list<SpatialIndex<3, UpLo::Lo, Frame::Inertial>>>
      Tl(used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Tl));

  // \f$L_{i} = R^{j} * S_{j} * T_{i}\f$
  const decltype(Tl) Li_from_Jji =
      TensorExpressions::evaluate<ti_i>(Ru(ti_J) * Sl(ti_j) * Tl(ti_i));
  // \f$L_{i} = R^{j} * T_{i} * S_{j}\f$
  const decltype(Tl) Li_from_Jij =
      TensorExpressions::evaluate<ti_i>(Ru(ti_J) * Tl(ti_i) * Sl(ti_j));
  // \f$L_{i} = T_{i} * S_{j} * R^{j}\f$
  const decltype(Tl) Li_from_ijJ =
      TensorExpressions::evaluate<ti_i>(Tl(ti_i) * Sl(ti_j) * Ru(ti_J));

  for (size_t i = 0; i < 3; i++) {
    DataType expected_product = make_with_value<DataType>(used_for_size, 0.0);
    for (size_t j = 0; j < 3; j++) {
      expected_product += (Ru.get(j) * Sl.get(j) * Tl.get(i));
    }
    CHECK(Li_from_Jji.get(i) == expected_product);
    CHECK(Li_from_Jij.get(i) == expected_product);
    CHECK(Li_from_ijJ.get(i) == expected_product);
  }

  using T_index = tmpl::front<typename decltype(Tl)::index_list>;
  using G_index = SpatialIndex<3, UpLo::Up, Frame::Inertial>;

  Tensor<DataType, Symmetry<2, 1>, index_list<G_index, G_index>> Guu(
      used_for_size);
  assign_unique_values_to_tensor(make_not_null(&Guu));

  // \f$L_{i}{}^{k} = S_{j} * T_{i} * G^{jk}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_Sj_Ti_GJK = TensorExpressions::evaluate<ti_i, ti_K>(
          Sl(ti_j) * Tl(ti_i) * Guu(ti_J, ti_K));
  // \f$L^{k}{}_{i} = S_{j} * T_{i} * G^{jk}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_Sj_Ti_GJK = TensorExpressions::evaluate<ti_K, ti_i>(
          Sl(ti_j) * Tl(ti_i) * Guu(ti_J, ti_K));
  // \f$L_{i}{}^{k} = S_{j} *  G^{jk} * T_{i}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_Sj_GJK_Ti = TensorExpressions::evaluate<ti_i, ti_K>(
          Sl(ti_j) * Guu(ti_J, ti_K) * Tl(ti_i));
  // \f$L^{k}{}_{i} = S_{j} *  G^{jk} * T_{i}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_Sj_GJK_Ti = TensorExpressions::evaluate<ti_K, ti_i>(
          Sl(ti_j) * Guu(ti_J, ti_K) * Tl(ti_i));
  // \f$L_{i}{}^{k} = T_{i} * S_{j} * G^{jk}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_Ti_Sj_GJK = TensorExpressions::evaluate<ti_i, ti_K>(
          Tl(ti_i) * Sl(ti_j) * Guu(ti_J, ti_K));
  // \f$L^{k}{}_{i} = T_{i} * S_{j} * G^{jk}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_Ti_Sj_GJK = TensorExpressions::evaluate<ti_K, ti_i>(
          Tl(ti_i) * Sl(ti_j) * Guu(ti_J, ti_K));
  // \f$L_{i}{}^{k} = T_{i} * G^{jk} * S_{j}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_Ti_GJK_Sj = TensorExpressions::evaluate<ti_i, ti_K>(
          Tl(ti_i) * Guu(ti_J, ti_K) * Sl(ti_j));
  // \f$L^{k}{}_{i} = T_{i} * G^{jk} * S_{j}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_Ti_GJK_Sj = TensorExpressions::evaluate<ti_K, ti_i>(
          Tl(ti_i) * Guu(ti_J, ti_K) * Sl(ti_j));
  // \f$L_{i}{}^{k} = G^{jk} * S_{j} * T_{i}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_GJK_Sj_Ti = TensorExpressions::evaluate<ti_i, ti_K>(
          Guu(ti_J, ti_K) * Sl(ti_j) * Tl(ti_i));
  // \f$L^{k}{}_{i} = G^{jk} * S_{j} * T_{i}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_GJK_Sj_Ti = TensorExpressions::evaluate<ti_K, ti_i>(
          Guu(ti_J, ti_K) * Sl(ti_j) * Tl(ti_i));
  // \f$L_{i}{}^{k} = G^{jk} * T_{i} * S_{j}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<T_index, G_index>>
      LiK_from_GJK_Ti_Sj = TensorExpressions::evaluate<ti_i, ti_K>(
          Guu(ti_J, ti_K) * Tl(ti_i) * Sl(ti_j));
  // \f$L^{k}{}_{i} = G^{jk} * T_{i} * S_{j}\f$
  const Tensor<DataType, Symmetry<2, 1>, index_list<G_index, T_index>>
      LKi_from_GJK_Ti_Sj = TensorExpressions::evaluate<ti_K, ti_i>(
          Guu(ti_J, ti_K) * Tl(ti_i) * Sl(ti_j));

  for (size_t k = 0; k < G_index::dim; k++) {
    for (size_t i = 0; i < T_index::dim; i++) {
      DataType expected_product =
          make_with_value<DataType>(used_for_size, 0.0);
      for (size_t j = 0; j < G_index::dim; j++) {
        expected_product += (Sl.get(j) * Tl.get(i) * Guu.get(j, k));
      }
      CHECK(LiK_from_Sj_Ti_GJK.get(i, k) == expected_product);
      CHECK(LKi_from_Sj_Ti_GJK.get(k, i) == expected_product);
      CHECK(LiK_from_Sj_GJK_Ti.get(i, k) == expected_product);
      CHECK(LKi_from_Sj_GJK_Ti.get(k, i) == expected_product);
      CHECK(LiK_from_Ti_Sj_GJK.get(i, k) == expected_product);
      CHECK(LKi_from_Ti_Sj_GJK.get(k, i) == expected_product);
      CHECK(LiK_from_Ti_GJK_Sj.get(i, k) == expected_product);
      CHECK(LKi_from_Ti_GJK_Sj.get(k, i) == expected_product);
      CHECK(LiK_from_GJK_Sj_Ti.get(i, k) == expected_product);
      CHECK(LKi_from_GJK_Sj_Ti.get(k, i) == expected_product);
      CHECK(LiK_from_GJK_Ti_Sj.get(i, k) == expected_product);
      CHECK(LKi_from_GJK_Ti_Sj.get(k, i) == expected_product);
    }
  }
}

template <typename DataType>
void test_products(const DataType& used_for_size) noexcept {
  test_outer_product_double(used_for_size);
  test_outer_product_rank_0_operand(used_for_size);
  test_outer_product_rank_1_operand(used_for_size);
  test_outer_product_rank_2x2_operands(used_for_size);
  test_outer_product_rank_0x1x2_operands(used_for_size);

  test_inner_product_rank_1x1_operands(used_for_size);
  test_inner_product_rank_2x2_operands(used_for_size);

  test_two_term_inner_outer_product(used_for_size);
  test_three_term_inner_outer_product(used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.Product",
                  "[DataStructures][Unit]") {
  test_products(std::numeric_limits<double>::signaling_NaN());
  test_products(DataVector(5, std::numeric_limits<double>::signaling_NaN()));
}
