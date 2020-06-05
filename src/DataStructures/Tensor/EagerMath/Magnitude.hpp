// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// @{
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

template <typename DataType, typename Index>
void magnitude(
    const gsl::not_null<Scalar<DataType>*> magnitude,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector) noexcept {
  destructive_resize_components(magnitude, get_size(get<0>(vector)));
  dot_product(magnitude, vector, vector);
  get(*magnitude) = sqrt(get(*magnitude));
}
// @}

// @{
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
  Scalar<DataType> local_magnitude{get_size(get<0>(vector))};
  magnitude(make_not_null(&local_magnitude), vector, metric);
  return local_magnitude;
}

template <typename DataType, typename Index>
void magnitude(
    const gsl::not_null<Scalar<DataType>*> magnitude,
    const Tensor<DataType, Symmetry<1>, index_list<Index>>& vector,
    const Tensor<DataType, Symmetry<1, 1>,
                 index_list<change_index_up_lo<Index>,
                            change_index_up_lo<Index>>>& metric) noexcept {
  dot_product(magnitude, vector, vector, metric);
  get(*magnitude) = sqrt(get(*magnitude));
}
// @}

// @{
/// \ingroup TensorGroup
/// \brief Compute square root of the Euclidean magnitude of a rank-0 tensor
///
/// \details
/// Computes the square root of the absolute value of the scalar.
template <typename DataType>
Scalar<DataType> sqrt_magnitude(const Scalar<DataType>& input) noexcept {
  return Scalar<DataType>{sqrt(abs(get(input)))};
}

template <typename DataType>
void sqrt_magnitude(const gsl::not_null<Scalar<DataType>*> sqrt_magnitude,
                    const Scalar<DataType>& input) noexcept {
  destructive_resize_components(sqrt_magnitude, get_size(get(input)));
  get(*sqrt_magnitude) = sqrt(abs(get(input)));
}
// @}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// The magnitude of a (co)vector
///
/// \snippet Test_Magnitude.cpp magnitude_name
template <typename Tag>
struct Magnitude : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Magnitude(" + db::tag_name<Tag>() + ")";
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
  using return_type = typename base::type;
  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<return_type*>,
                           const typename Tag::type&) noexcept>(&magnitude);
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
  using return_type = typename base::type;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<return_type*>, const typename Tag::type&,
      const typename MetricTag::type&) noexcept>(&magnitude);
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
    return "Normalized(" + db::tag_name<Tag>() + ")";
  }
  using tag = Tag;
  using type = db::const_item_type<Tag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DataStructuresGroup
/// Normalizes the (co)vector represented by Tag
///
/// This tag inherits from `Tags::Normalized<Tag>`
template <typename Tag>
struct NormalizedCompute : Normalized<Tag>, db::ComputeTag {
  using base = Normalized<Tag>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> normalized_vector,
      const typename Tag::type& vector_in,
      const typename Magnitude<Tag>::type& magnitude) noexcept {
    destructive_resize_components(normalized_vector, get_size(get(magnitude)));
    *normalized_vector = vector_in;
    for (size_t d = 0; d < normalized_vector->index_dim(0); ++d) {
      normalized_vector->get(d) /= get(magnitude);
    }
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
  static std::string name() noexcept {
    return "Sqrt(" + db::tag_name<Tag>() + ")";
  }
  using return_type = Scalar<DataVector>;
  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<return_type*>, const typename Tag::type&) noexcept>(
      &sqrt_magnitude);
  using argument_tags = tmpl::list<Tag>;
};
}  // namespace Tags
