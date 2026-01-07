// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/GramSchmidtOrthonormalize.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"

template <typename DataType, typename Index, size_t NumVectors>
void gram_schmidt_orthonormalize(
    const std::array<
        gsl::not_null<Tensor<DataType, Symmetry<1>, index_list<Index>>*>,
        NumVectors>& basis,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) {
  Scalar<DataType> buffer{};
  // Keep track of the sign of each basis vector for spacetime metrics. Avoid
  // unnecessary memory allocation for spatial metrics by using simple sign type
  // (will not be used, so the type doesn't matter).
  using sign_dtype =
      std::conditional_t<Index::index_type == IndexType::Spacetime, DataType,
                         int>;
  std::array<sign_dtype, NumVectors> basis_signs{};
  const auto normalize = [&metric, &norm = buffer](
                             auto& v, [[maybe_unused]] auto& v_sign) {
    dot_product(make_not_null(&norm), v, v, metric);
    if constexpr (Index::index_type == IndexType::Spacetime) {
      v_sign = sgn(get(norm));
      get(norm) = 1.0 / sqrt(abs(get(norm)));
    } else {
      get(norm) = 1.0 / sqrt(get(norm));
    }
    for (size_t k = 0; k < v.size(); ++k) {
      v.get(k) *= get(norm);
    }
  };
  // Normalize the first vector.
  normalize(*basis.at(0), basis_signs.at(0));
  // Orthogonalize the remaining vectors
  auto& projection = buffer;
  for (size_t i = 1; i < basis.size(); ++i) {
    auto& v = *basis.at(i);
    for (size_t j = 0; j < i; ++j) {
      auto& w = *basis.at(j);
      dot_product(make_not_null(&projection), v, w, metric);
      // w is already normalized, but need to account for sign in spacetime case
      if constexpr (Index::index_type == IndexType::Spacetime) {
        get(projection) *= basis_signs.at(j);
      }
      for (size_t k = 0; k < v.size(); ++k) {
        v.get(k) -= get(projection) * w.get(k);
      }
    }
    normalize(v, basis_signs.at(i));
  }
}

// Instantiate for double and DataVector
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define UPLO(data) BOOST_PP_TUPLE_ELEM(3, data)
#define INDEXTYPE(data) BOOST_PP_TUPLE_ELEM(4, data)
#define NUMVEC(data) BOOST_PP_TUPLE_ELEM(5, data)

#define INSTANTIATION(r, data)                                              \
  template void gram_schmidt_orthonormalize(                                \
      const std::array<                                                     \
          gsl::not_null<Tensor<                                             \
              DTYPE(data), Symmetry<1>,                                     \
              index_list<Tensor_detail::TensorIndexType<                    \
                  DIM(data), UPLO(data), FRAME(data), INDEXTYPE(data)>>>*>, \
          NUMVEC(data)>& basis,                                             \
      const Tensor<                                                         \
          DTYPE(data), Symmetry<1, 1>,                                      \
          index_list<                                                       \
              change_index_up_lo<Tensor_detail::TensorIndexType<            \
                  DIM(data), UPLO(data), FRAME(data), INDEXTYPE(data)>>,    \
              change_index_up_lo<Tensor_detail::TensorIndexType<            \
                  DIM(data), UPLO(data), FRAME(data), INDEXTYPE(data)>>>>&  \
          metric);

GENERATE_INSTANTIATIONS(INSTANTIATION, (double), (3), (Frame::Inertial),
                        (UpLo::Lo, UpLo::Up),
                        (IndexType::Spacetime, IndexType::Spatial), (3))

#undef DTYPE
#undef DIM
#undef FRAME
#undef UPLO
#undef INDEXTYPE
#undef NUMVEC
#undef INSTANTIATION
