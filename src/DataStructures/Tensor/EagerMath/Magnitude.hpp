// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup TensorGroup
 * \brief Compute the Euclidean magnitude of a rank-1 tensor
 *
 * \details
 * Computes the square root of the sum of the squares of the components of
 * the rank-1 tensor.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector) noexcept {
  return Scalar<DataType>{sqrt(get(dot_product(vector, vector)))};
}

/*!
 * \ingroup TensorGroup
 * \brief Compute the magnitude of a rank-1 tensor
 *
 * \details
 * Returns the square root of the input tensor contracted twice with the given
 * metric.
 */
template <typename DataType, typename Index>
Scalar<DataType> magnitude(
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) noexcept {
  return Scalar<DataType>{sqrt(get(dot_product(vector, vector, metric)))};
}

/// \ingroup TensorGroup
/// \brief Compute square root of the Euclidean magnitude of a rank-0 tensor
///
/// \details
/// Computes the square root of the absolute value of the scalar.
template <typename DataType>
Scalar<DataType> sqrt_magnitude(const Scalar<DataType>& input) noexcept {
  return Scalar<DataType>{sqrt(abs(get(input)))};
}

/// @{
/// \ingroup TensorGroup
/// \brief Compute $\\ell^2$ (Frobenius) norm of a tensor
///
/// \details
/// Computes the Frobenius norm of a given tensor of arbitrary rank.
template <typename DataType, typename Symm, typename IndexList>
Scalar<DataType> l2_norm(
    const Tensor<DataType, Symm, IndexList>& tensor) noexcept {
  auto& input = const_cast<Tensor<DataType, Symm, IndexList>&>(tensor);
  DataType sum(tensor.begin()->size(), 0.);
  for (auto it = input.begin(); it != input.end(); ++it) {
    sum += input.multiplicity(it) * square(*it);
  }
  return Scalar<DataType>{sqrt(sum)};
}
template <typename DataType, typename Symm, typename IndexList>
double l2_norm(const Tensor<DataType, Symm, IndexList>& tensor) noexcept {
  auto& input = const_cast<Tensor<DataType, Symm, IndexList>&>(tensor);
  DataType sum(tensor.begin()->size(), 0.);
  for (auto it = input.begin(); it != input.end(); ++it) {
    sum += input.multiplicity(it) * square(*it);
  }
  using PlusSquare = funcl::Plus<funcl::Identity, funcl::Square<>>;
  return alg::accumulate(sqrt(sum), 0., PlusSquare{});
}
/// @}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The magnitude of a (co)vector
///
/// \snippet Test_Magnitude.cpp magnitude_name
template <typename Tag>
struct Magnitude : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Magnitude(" + Tag::name() + ")";
  }
  using tag = Tag;
  using type = Scalar<DataVector>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The Euclidean magnitude of a (co)vector
///
/// This tag inherits from `Tags::Magnitude<Tag>`
template <typename Tag>
struct EuclideanMagnitude : Magnitude<Tag>, db::ComputeTag {
  using base = Magnitude<Tag>;
  static constexpr Scalar<DataVector> (*function)(const db::item_type<Tag>&) =
      magnitude;
  using argument_tags = tmpl::list<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The magnitude of a (co)vector with respect to a specific metric
///
/// This tag inherits from `Tags::Magnitude<Tag>`
template <typename Tag, typename MetricTag>
struct NonEuclideanMagnitude : Magnitude<Tag>, db::ComputeTag {
  using base = Magnitude<Tag>;
  static constexpr Scalar<DataVector> (*function)(
      const db::item_type<Tag>&, const db::item_type<MetricTag>&) = magnitude;
  using argument_tags = tmpl::list<Tag, MetricTag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The normalized (co)vector represented by Tag
///
/// \snippet Test_Magnitude.cpp normalized_name
template <typename Tag>
struct Normalized : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Normalized(" + Tag::name() + ")";
  }
  using tag = Tag;
  using type = db::item_type<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// Normalizes the (co)vector represented by Tag
///
/// This tag inherits from `Tags::Normalized<Tag>`
template <typename Tag>
struct NormalizedCompute : Normalized<Tag>, db::ComputeTag {
  using base = Normalized<Tag>;
  static constexpr auto function(
      const db::item_type<Tag>&
          vector_in,  // Compute items need to take const references
      const db::item_type<Magnitude<Tag>>& magnitude) noexcept {
    auto vector = vector_in;
    for (size_t d = 0; d < vector.index_dim(0); ++d) {
      vector.get(d) /= get(magnitude);
    }
    return vector;
  }
  using argument_tags = tmpl::list<Tag, Magnitude<Tag>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The square root of a scalar
///
/// \snippet Test_Magnitude.cpp sqrt_name
template <typename Tag>
struct Sqrt : db::ComputeTag {
  static std::string name() noexcept { return "Sqrt(" + Tag::name() + ")"; }
  static constexpr Scalar<DataVector> (*function)(const db::item_type<Tag>&) =
      sqrt_magnitude;
  using argument_tags = tmpl::list<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The L2 norm of a tensor
///
/// \snippet Test_Magnitude.cpp sqrt_name
template <typename Tag, bool Accumulate = false>
struct L2Norm : db::ComputeTag {
  static std::string name() noexcept { return "L2Norm(" + Tag::name() + ")"; }
  static constexpr tmpl::conditional_t<Accumulate, double, Scalar<DataVector>> (
      *function)(const db::item_type<Tag>&) = l2_norm;
  using argument_tags = tmpl::list<Tag>;
};
}  // namespace Tags
