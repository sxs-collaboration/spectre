// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TensorStructure<Symmetry, Indices...>

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/IntelDetails.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TensorGroup
/// Contains details of the implementation of Tensor
namespace Tensor_detail {
namespace detail {
template <typename = void>
SPECTRE_ALWAYS_INLINE constexpr auto compute_collapsed_index_impl() noexcept {
  return 0;
}

template <typename>
SPECTRE_ALWAYS_INLINE constexpr auto compute_collapsed_index_impl(
    std::size_t i) noexcept {
  return i;
}

template <typename IndexList, typename... I>
inline constexpr std::size_t compute_collapsed_index_impl(std::size_t I0,
                                                          I... i) noexcept {
  return I0 +
         tmpl::at<IndexList, tmpl::size_t<tmpl::size<IndexList>::value -
                                          sizeof...(I) - 1>>::dim *
             compute_collapsed_index_impl<IndexList>(i...);
}

template <size_t NumIndices>
struct ComponentNameImpl {
  template <typename Structure, typename T>
  static std::string apply(
      const std::array<T, NumIndices>& tensor_index,
      const std::array<std::string, NumIndices>& axis_labels) {
    const size_t storage_index = Structure::get_storage_index(tensor_index);
    std::array<std::string, Structure::rank()> labels = axis_labels;
    constexpr auto index_dim = Structure::dims();
    for (size_t i = 0; i < Structure::rank(); ++i) {
      if (gsl::at(labels, i).length() == 0) {
        if (gsl::at(Structure::index_types(), i) == IndexType::Spacetime) {
          switch (gsl::at(index_dim, i)) {
            case 2:
              gsl::at(labels, i) = "tx";
              break;
            case 3:
              gsl::at(labels, i) = "txy";
              break;
            case 4:
              gsl::at(labels, i) = "txyz";
              break;
            default:
              ERROR("Tensor dim["
                    << i << "] must be 1,2,3, or 4 for default axis_labels. "
                            "Either pass a string or extend the function.");
          }
        } else {
          switch (gsl::at(index_dim, i)) {
            case 1:
              gsl::at(labels, i) = "x";
              break;
            case 2:
              gsl::at(labels, i) = "xy";
              break;
            case 3:
              gsl::at(labels, i) = "xyz";
              break;
            default:
              ERROR("Tensor dim["
                    << i << "] must be 1,2, or 3 for default axis_labels. "
                            "Either pass a string or extend the function.");
          }
        }
      } else {
        if (gsl::at(axis_labels, i).length() != gsl::at(index_dim, i)) {
          ERROR("Dimension mismatch: Tensor has dim = "
                << gsl::at(index_dim, i) << ", but you specified "
                << gsl::at(axis_labels, i).length() << " different labels in "
                << gsl::at(axis_labels, i));
        }
      }
    }
    // Create string labeling get_tensor_index
    std::stringstream ss;
    const auto& canonical_tensor_index =
        Structure::get_canonical_tensor_index(storage_index);
    for (size_t r = 0; r < Structure::rank(); ++r) {
      ss << gsl::at(labels, r)[gsl::at(canonical_tensor_index, r)];
    }
    return ss.str();
  }
};

template <>
struct ComponentNameImpl<0> {
  template <typename Structure, typename T>
  static std::string apply(const std::array<T, 0>& /*tensor_index*/,
                           const std::array<std::string, 0>& /*axis_labels*/) {
    return "Scalar";
  }
};
}  // namespace detail

template <typename T, typename S, std::size_t Rank>
inline constexpr std::size_t compute_collapsed_index(
    const std::array<T, Rank>& tensor_index, const std::array<S, Rank> dims,
    const size_t i = 0) noexcept {
  static_assert(tt::is_integer_v<T>,
                "The tensor index array must hold integer types.");
  static_assert(tt::is_integer_v<S>, "The dims array must hold integer types.");
  return i < Rank ? (gsl::at(tensor_index, i) +
                     gsl::at(dims, i) *
                         compute_collapsed_index(tensor_index, dims, i + 1))
                  : 0;
}

template <typename IndexList, typename... I,
          Requires<cpp17::conjunction_v<tt::is_integer<I>...>> = nullptr>
SPECTRE_ALWAYS_INLINE constexpr size_t compute_collapsed_index(
    I... i) noexcept {
  static_assert(sizeof...(I) == tmpl::size<IndexList>::value,
                "The number of tensor indices passed to "
                "compute_collapsed_index does not match the rank of the "
                "tensor.");
  return static_cast<size_t>(detail::compute_collapsed_index_impl<IndexList>(
      static_cast<std::size_t>(i)...));
}

/// \ingroup TensorGroup
/// A lookup table between each tensor_index and storage_index
///
/// 1. tensor_index: (a, b, c,...). There are Dim^rank tensor_index's
/// 2. collapsed_index: a + Dim * (b + Dim * (c + ...)), there are Dim^rank
///                     unique collapsed indices and there is a 1-1 map between
///                     a tensor_index and a collapsed_index.
/// 3. storage_index: index into the storage vector of the Tensor. This depends
///                   on symmetries of the tensor, rank and dimensionality.
///                   There are size storage_index's.
/// \tparam Symm the symmetry of the Tensor
/// \tparam Indices list of tensor_index's giving the dimensionality and frame
/// of the index
template <typename Symm, typename... Indices>
struct Structure {
  static_assert(
      TensorMetafunctions::check_index_symmetry<Symm,
                                                tmpl::list<Indices...>>::value,
      "Cannot construct a Tensor with a symmetric pair that are not the same.");
  static_assert(tmpl::size<Symm>::value == sizeof...(Indices),
                "The number of indices in Symmetry do not match the number of "
                "indices given to the Structure.");
  static_assert(
      cpp17::conjunction<tt::is_tensor_index_type<Indices>...>::value,
      "All Indices passed to Structure must be of type TensorIndexType.");

  using index_list = tmpl::list<Indices...>;
  using num_of_components =
      TensorMetafunctions::number_of_components<Symm, index_list>;
  using NumberOfIndependentComponents =
      TensorMetafunctions::independent_components<Symm, index_list>;

#ifdef __INTEL_COMPILER
  // The Intel compiler is embarrassingly terrible at compiling metaprograms,
  // possibly because of a lack of memoization, but without having access to
  // their compiler source we cannot be sure. Unfortunately ICC is also terrible
  // at supporting constant expressions and so moving these computations to
  // constant expressions is also not feasible (it's been attempted). As a
  // result we do the computations of the arrays at compile time when using GCC
  // or Clang and at runtime when using Intel. It should be possible to later
  // make the arrays static so they are only computed at most once during
  // evolution. However, if Tensor has the structure as a static member variable
  // then this is irrelevant anyway.
  //
  // ICC chokes on the computation of the collapsed_to_storage_ array.
  // Specifically it cannot handle the IncrementTensorIndex call. We've tested
  // several other implementations of this metafunction but they all result in
  // the same time with GCC and Clang but impossibly long compilation with ICC.
  const std::array<size_t,
                   number_of_components<Symm, tmpl::list<Indices...>>::value>
      collapsed_to_storage_ = Tensor_detail::compute_collapsed_to_storage<
          Symm, tmpl::list<Indices...>,
          number_of_components<Symm, index_list>>();
  const std::array<
      std::array<size_t, sizeof...(Indices) == 0 ? 1 : sizeof...(Indices)>,
      IndependentComponents<Symm, tmpl::list<Indices...>>::value>
      storage_to_tensor_ = Tensor_detail::compute_storage_to_tensor<
          Symm, tmpl::list<Indices...>,
          IndependentComponents<Symm, index_list>>(collapsed_to_storage_);
  const std::array<size_t,
                   IndependentComponents<Symm, tmpl::list<Indices...>>::value>
      multiplicity_ = Tensor_detail::compute_multiplicity<
          IndependentComponents<Symm, tmpl::list<Indices...>>>(
          collapsed_to_storage_);
#else
  using collapsed_to_storage_list =
      TensorMetafunctions::compute_collapsed_to_storage<index_list, Symm,
                                                        num_of_components>;
  using storage_to_tensor_index_list =
      TensorMetafunctions::compute_storage_to_tensor<
          Symm, index_list, collapsed_to_storage_list,
          NumberOfIndependentComponents>;

  using multiplicity_list =
      TensorMetafunctions::compute_multiplicity<collapsed_to_storage_list,
                                                NumberOfIndependentComponents>;
#endif

 public:
  SPECTRE_ALWAYS_INLINE static constexpr size_t rank() noexcept {
    return sizeof...(Indices);
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t size() noexcept {
    return NumberOfIndependentComponents::value;
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t
  number_of_components() noexcept {
    return num_of_components::value;
  }

  // Retrieves the dimensionality of the I'th index
  template <int I>
  SPECTRE_ALWAYS_INLINE static constexpr size_t dim() noexcept {
    static_assert(sizeof...(Indices),
                  "A scalar does not have any indices from which you can "
                  "retrieve the dimensionality.");
    return tmpl::at<index_list, tmpl::int32_t<I>>::value;
  }

  static constexpr std::array<size_t, sizeof...(Indices)> dims() noexcept {
    return make_array_from_list<
        tmpl::conditional_t<sizeof...(Indices) != 0, index_list, size_t>>();
  }

  SPECTRE_ALWAYS_INLINE static constexpr std::array<int, sizeof...(Indices)>
  symmetries() noexcept {
    return make_array_from_list<
        tmpl::conditional_t<0 != sizeof...(Indices), Symm, int>>();
  }

  SPECTRE_ALWAYS_INLINE static constexpr std::array<IndexType,
                                                    sizeof...(Indices)>
  index_types() noexcept {
    return std::array<IndexType, sizeof...(Indices)>{{Indices::index_type...}};
  }

  /// Return array of the valence of each index
  SPECTRE_ALWAYS_INLINE static constexpr std::array<UpLo, sizeof...(Indices)>
  index_valences() noexcept {
    return std::array<UpLo, sizeof...(Indices)>{{Indices::ul...}};
  }

  /// Return array of the frame of each index
  SPECTRE_ALWAYS_INLINE static constexpr auto index_frames() noexcept {
    return std::tuple<typename Indices::Frame...>{};
  }

  /*!
   * \brief Get the canonical tensor_index array
   *
   * \details
   * For a symmetric tensor \f$T_{(ab)}\f$ with an associated symmetry list
   * `Symmetry<1, 1>`, this will return, e.g. `{{3, 2}}` rather than `{{2, 3}}`
   * for that particular index.
   * Note that this ordering is implementation defined.
   */
  template <size_t Rank = sizeof...(Indices),
            std::enable_if_t<Rank != 0>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  get_canonical_tensor_index(const size_t storage_index) noexcept {
    return gsl::at(make_array_from_list<storage_to_tensor_index_list>(),
                   storage_index);
  }
  template <size_t Rank = sizeof...(Indices),
            std::enable_if_t<Rank == 0>* = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, 0>
  get_canonical_tensor_index(const size_t /*storage_index*/) noexcept {
    return std::array<size_t, 0>{};
  }

  /// Get storage_index
  /// \param args comma separated list of the index to return
  template <typename... N>
  SPECTRE_ALWAYS_INLINE static constexpr std::size_t get_storage_index(
      const N... args) noexcept {
    static_assert(sizeof...(Indices) == sizeof...(N),
                  "the number arguments must be equal to rank_");
    return gsl::at(
        make_array_from_list<collapsed_to_storage_list>(),
        compute_collapsed_index<index_list>(static_cast<size_t>(args)...));
  }
  /// Get storage_index
  /// \param tensor_index the tensor_index of which to get the storage_index
  template <typename I>
  SPECTRE_ALWAYS_INLINE static constexpr std::size_t get_storage_index(
      const std::array<I, sizeof...(Indices)>& tensor_index) noexcept {
    return gsl::at(make_array_from_list<collapsed_to_storage_list>(),
                   compute_collapsed_index(tensor_index, Structure::dims()));
  }

  template <int... N, Requires<(sizeof...(N) > 0)> = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr std::size_t
  get_storage_index() noexcept {
    static_assert(sizeof...(Indices) == sizeof...(N),
                  "the number arguments must be equal to rank_");
    return tmpl::at<
        collapsed_to_storage_list,
        TensorMetafunctions::compute_collapsed_index<
            TensorMetafunctions::canonicalize_tensor_index<
                Symm, index_list, tmpl::integral_list<std::size_t, N...>>,
            index_list>>::value;
  }

  /// Get the multiplicity of the storage_index
  /// \param storage_index the storage_index of which to get the multiplicity
  SPECTRE_ALWAYS_INLINE static constexpr size_t multiplicity(
      const size_t storage_index) noexcept {
    return gsl::at(make_array_from_list<multiplicity_list>(), storage_index);
  }

  /// Get the array of collapsed index to storage_index
  SPECTRE_ALWAYS_INLINE static constexpr std::array<int,
                                                    number_of_components()>&
  collapsed_to_storage() noexcept {
    return make_array_from_list<collapsed_to_storage_list>();
  }

  /// Get the storage_index for the specified collapsed index
  SPECTRE_ALWAYS_INLINE static constexpr int collapsed_to_storage(
      const size_t i) noexcept {
    return gsl::at(make_array_from_list<collapsed_to_storage_list>(), i);
  }

  /// Get the array of tensor_index's corresponding to the storage_index's.
  SPECTRE_ALWAYS_INLINE static constexpr const std::array<
      std::array<size_t, sizeof...(Indices) == 0 ? 1 : sizeof...(Indices)>,
      TensorMetafunctions::independent_components<Symm, index_list>::value>&
  storage_to_tensor_index() noexcept {
    return make_array_from_list<tmpl::conditional_t<
        sizeof...(Indices) == 0, tmpl::list<tmpl::list<tmpl::size_t<0>>>,
        storage_to_tensor_index_list>>();
  }

  template <typename T>
  SPECTRE_ALWAYS_INLINE static std::string component_name(
      const std::array<T, rank()>& tensor_index,
      const std::array<std::string, rank()>& axis_labels) {
    return detail::ComponentNameImpl<sizeof...(
        Indices)>::template apply<Structure>(tensor_index, axis_labels);
  }
};
}  // namespace Tensor_detail
