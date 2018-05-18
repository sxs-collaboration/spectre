// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TensorStructure<Symmetry, Indices...>

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Symmetry.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Array.hpp"  // IWYU pragma: export
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace Tensor_detail {
constexpr size_t number_of_independent_components(
    const std::array<int, 0>& /*symm*/,
    const std::array<size_t, 0>& /*dims*/) noexcept {
  return 1;
}

constexpr size_t number_of_independent_components(
    const std::array<int, 1>& /*symm*/,
    const std::array<size_t, 1>& dims) noexcept {
  return dims[0];
}

template <size_t Size>
constexpr size_t number_of_independent_components(
    const std::array<int, Size>& symm,
    const std::array<size_t, Size>& dims) noexcept {
  size_t max_element = 0;
  for (size_t i = 0; i < Size; ++i) {
    // clang-tidy: internal STL and gls::at (don't need it in constexpr)
    assert(symm[i] > 0);  // NOLINT
    max_element =
        std::max(static_cast<size_t>(ce_abs(symm[i])), max_element);  // NOLINT
  }
  assert(max_element > 0);  // NOLINT
  size_t total_independent_components = 1;
  for (size_t symm_index = 1; symm_index <= max_element; ++symm_index) {
    size_t number_of_indices_with_symm = 0;
    size_t dim_of_index = 0;
    for (size_t i = 0; i < Size; ++i) {
      if (static_cast<size_t>(symm[i]) == symm_index) {  // NOLINT
        ++number_of_indices_with_symm;
        dim_of_index = dims[i];  // NOLINT
      }
    }
    assert(dim_of_index > 0);                 // NOLINT
    assert(number_of_indices_with_symm > 0);  // NOLINT
    if (dim_of_index - 1 > number_of_indices_with_symm) {
      total_independent_components *=
          falling_factorial(dim_of_index + number_of_indices_with_symm - 1,
                            number_of_indices_with_symm) /
          factorial(number_of_indices_with_symm);
    } else {
      total_independent_components *=
          falling_factorial(dim_of_index + number_of_indices_with_symm - 1,
                            dim_of_index - 1) /
          factorial(dim_of_index - 1);
    }
  }
  return total_independent_components;
}

template <size_t Size>
constexpr size_t number_of_components(
    const std::array<size_t, Size>& dims) noexcept {
  size_t number = 1;
  for (size_t i = 0; i < Size; ++i) {
    // clang-tidy: use gsl::at
    number *= dims[i];  // NOLINT
  }
  return number;
}

template <typename T, typename S, size_t Size>
constexpr void increment_tensor_index(cpp17::array<T, Size>& tensor_index,
                                      const cpp17::array<S, Size>& dims) {
  for (size_t i = 0; i < Size; ++i) {
    if (++tensor_index[i] < static_cast<T>(dims[i])) {
      return;
    }
    tensor_index[i] = 0;
  }
}

// index_to_swap_with takes the last two arguments as opposed to just one of
// them so that when the max constexpr steps is reached on clang it is reached
// in this function rather than in array.
template <size_t Rank>
constexpr size_t index_to_swap_with(
    const cpp17::array<size_t, Rank>& tensor_index,
    const cpp17::array<int, Rank>& sym, size_t index_to_swap_with,
    const size_t current_index) noexcept {
  // If you encounter infinite loop compilation errors here you are
  // constructing very large Tensor's. If you are sure Tensor is
  // the correct data structure you can extend the compiler limit
  // by passing the flag -fconstexpr-steps=<SOME LARGER VALUE>
  while (true) {  // See source code comment on line above this one for fix
    if (Rank == index_to_swap_with) {
      return current_index;
    } else if (tensor_index[current_index] <
                   tensor_index[index_to_swap_with] and
               sym[current_index] == sym[index_to_swap_with]) {
      return index_to_swap_with;
    }
    index_to_swap_with++;
  }
}

template <size_t Size, size_t SymmSize>
constexpr cpp17::array<size_t, Size> canonicalize_tensor_index(
    cpp17::array<size_t, Size> tensor_index,
    const cpp17::array<int, SymmSize>& symm) noexcept {
  for (size_t i = 0; i < Size; ++i) {
    const size_t temp = tensor_index[i];
    const size_t swap = index_to_swap_with(tensor_index, symm, i, i);
    tensor_index[i] = tensor_index[swap];
    tensor_index[swap] = temp;
  }
  return tensor_index;
}

template <size_t Rank>
constexpr size_t compute_collapsed_index(
    const cpp17::array<size_t, Rank>& tensor_index,
    const cpp17::array<size_t, Rank> dims) noexcept {
  size_t collapsed_index = 0;
  for (size_t i = Rank - 1; i < Rank; --i) {
    collapsed_index = tensor_index[i] + dims[i] * collapsed_index;
  }
  return collapsed_index;
}

template <typename Symm, size_t NumberOfComponents,
          Requires<tmpl::size<Symm>::value != 0> = nullptr>
constexpr cpp17::array<size_t, NumberOfComponents> compute_collapsed_to_storage(
    const cpp17::array<size_t, tmpl::size<Symm>::value>&
        index_dimensions) noexcept {
  cpp17::array<size_t, NumberOfComponents> collapsed_to_storage{};
  auto tensor_index =
      convert_to_cpp17_array(make_array<tmpl::size<Symm>::value>(size_t{0}));
  size_t count{0};
  for (auto& current_storage_index : collapsed_to_storage) {
    // Compute canonical tensor_index, which, for symmetric get_tensor_index is
    // in decreasing numerical order, e.g. (3,2) rather than (2,3).
    const auto canonical_tensor_index = canonicalize_tensor_index(
        tensor_index, make_cpp17_array_from_list<Symm>());
    // If the tensor_index was already in the canonical form, then it must be a
    // new unique entry  and we add it to collapsed_to_storage_ as a new
    // integer, thus increasing the size_. Else, the StorageIndex has already
    // been determined so we look it up in the existing collapsed_to_storage
    // table.
    if (tensor_index == canonical_tensor_index) {
      current_storage_index = count;
      ++count;
    } else {
      current_storage_index = collapsed_to_storage[compute_collapsed_index(
          canonical_tensor_index, index_dimensions)];
    }
    // Move to the next tensor_index.
    increment_tensor_index(tensor_index, index_dimensions);
  }
  return collapsed_to_storage;
}

template <typename Symm, size_t NumberOfComponents,
          Requires<tmpl::size<Symm>::value == 0> = nullptr>
constexpr cpp17::array<size_t, 1> compute_collapsed_to_storage(
    const cpp17::array<
        size_t, tmpl::size<Symm>::value>& /*index_dimensions*/) noexcept {
  return cpp17::array<size_t, 1>{{0}};
}

template <typename Symm, size_t NumIndComps, size_t NumComps,
          Requires<(tmpl::size<Symm>::value > 0)> = nullptr>
constexpr cpp17::array<cpp17::array<size_t, tmpl::size<Symm>::value>,
                       NumIndComps>
compute_storage_to_tensor(
    const cpp17::array<size_t, NumComps>& collapsed_to_storage,
    const cpp17::array<size_t, tmpl::size<Symm>::value>&
        index_dimensions) noexcept {
  constexpr size_t rank = tmpl::size<Symm>::value;
  cpp17::array<cpp17::array<size_t, rank>, NumIndComps> storage_to_tensor{};
  cpp17::array<size_t, rank> tensor_index =
      convert_to_cpp17_array(make_array<rank>(size_t{0}));
  for (const auto& current_storage_index : collapsed_to_storage) {
    storage_to_tensor[current_storage_index] = canonicalize_tensor_index(
        tensor_index, make_cpp17_array_from_list<Symm>());
    increment_tensor_index(tensor_index, index_dimensions);
  }
  return storage_to_tensor;
}

template <typename Symm, size_t NumIndComps, size_t NumComps,
          Requires<(tmpl::size<Symm>::value == 0)> = nullptr>
constexpr cpp17::array<cpp17::array<int, 1>, NumIndComps>
compute_storage_to_tensor(
    const cpp17::array<size_t, NumComps>& /*collapsed_to_storage*/,
    const cpp17::array<size_t, tmpl::size<Symm>::value>&
    /*index_dimensions*/) noexcept {
  return cpp17::array<cpp17::array<int, 1>, 1>{{cpp17::array<int, 1>{{0}}}};
}

template <size_t NumIndComps, typename T, size_t NumComps>
constexpr cpp17::array<size_t, NumIndComps> compute_multiplicity(
    const cpp17::array<T, NumComps>& collapsed_to_storage) {
  cpp17::array<size_t, NumIndComps> multiplicity =
      convert_to_cpp17_array(make_array<NumIndComps>(size_t{0}));
  for (const auto& current_storage_index : collapsed_to_storage) {
    ++multiplicity[current_storage_index];
  }
  return multiplicity;
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
                    << i
                    << "] must be 1,2,3, or 4 for default axis_labels. "
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
                    << i
                    << "] must be 1,2, or 3 for default axis_labels. "
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
      TensorMetafunctions::check_index_symmetry_v<Symm, Indices...>,
      "Cannot construct a Tensor with a symmetric pair that are not the same.");
  static_assert(tmpl::size<Symm>::value == sizeof...(Indices),
                "The number of indices in Symmetry do not match the number of "
                "indices given to the Structure.");
  static_assert(
      tmpl2::flat_all_v<tt::is_tensor_index_type<Indices>::value...>,
      "All Indices passed to Structure must be of type TensorIndexType.");

  using index_list = tmpl::list<Indices...>;

  SPECTRE_ALWAYS_INLINE static constexpr size_t rank() noexcept {
    return sizeof...(Indices);
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t size() noexcept {
    constexpr auto number_of_independent_components =
        ::Tensor_detail::number_of_independent_components(
            make_array_from_list<
                tmpl::conditional_t<sizeof...(Indices) != 0, Symm, int>>(),
            make_array_from_list<tmpl::conditional_t<sizeof...(Indices) != 0,
                                                     index_list, size_t>>());
    return number_of_independent_components;
  }

  SPECTRE_ALWAYS_INLINE static constexpr size_t
  number_of_components() noexcept {
    constexpr auto number_of_components = ::Tensor_detail::number_of_components(
        make_array_from_list<tmpl::conditional_t<sizeof...(Indices) != 0,
                                                 index_list, size_t>>());
    return number_of_components;
  }

  static constexpr auto collapsed_to_storage_ =
      compute_collapsed_to_storage<Symm, number_of_components()>(
          make_cpp17_array_from_list<tmpl::conditional_t<
              sizeof...(Indices) == 0, size_t, index_list>>());
  static constexpr auto storage_to_tensor_ = compute_storage_to_tensor<Symm,
                                                                       size()>(
      collapsed_to_storage_,
      make_cpp17_array_from_list<
          tmpl::conditional_t<sizeof...(Indices) == 0, size_t, index_list>>());
  static constexpr auto multiplicity_ =
      compute_multiplicity<size()>(collapsed_to_storage_);

  // Retrieves the dimensionality of the I'th index
  template <int I>
  SPECTRE_ALWAYS_INLINE static constexpr size_t dim() noexcept {
    static_assert(sizeof...(Indices),
                  "A scalar does not have any indices from which you can "
                  "retrieve the dimensionality.");
    return tmpl::at<index_list, tmpl::int32_t<I>>::value;
  }

  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  dims() noexcept {
    constexpr auto dims = make_array_from_list<
        tmpl::conditional_t<sizeof...(Indices) != 0, index_list, size_t>>();
    return dims;
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
    constexpr auto storage_to_tensor = storage_to_tensor_;
    return gsl::at(storage_to_tensor, storage_index);
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
    constexpr auto collapsed_to_storage = collapsed_to_storage_;
    return gsl::at(
        collapsed_to_storage,
        compute_collapsed_index(
            canonicalize_tensor_index(
                cpp17::array<size_t, sizeof...(N)>{
                    {static_cast<size_t>(args)...}},
                make_cpp17_array_from_list<
                    tmpl::conditional_t<0 != sizeof...(Indices), Symm, int>>()),
            make_cpp17_array_from_list<tmpl::conditional_t<
                0 != sizeof...(Indices), index_list, size_t>>()));
  }
  /// Get storage_index
  /// \param tensor_index the tensor_index of which to get the storage_index
  template <typename I>
  SPECTRE_ALWAYS_INLINE static constexpr std::size_t get_storage_index(
      const std::array<I, sizeof...(Indices)>& tensor_index) noexcept {
    constexpr auto collapsed_to_storage = collapsed_to_storage_;
    return gsl::at(
        collapsed_to_storage,
        compute_collapsed_index(
            canonicalize_tensor_index(
                convert_to_cpp17_array(tensor_index),
                make_cpp17_array_from_list<
                    tmpl::conditional_t<0 != sizeof...(Indices), Symm, int>>()),
            make_cpp17_array_from_list<tmpl::conditional_t<
                0 != sizeof...(Indices), index_list, size_t>>()));
  }

  template <int... N, Requires<(sizeof...(N) > 0)> = nullptr>
  SPECTRE_ALWAYS_INLINE static constexpr std::size_t
  get_storage_index() noexcept {
    static_assert(sizeof...(Indices) == sizeof...(N),
                  "the number arguments must be equal to rank_");
    constexpr std::size_t storage_index =
        collapsed_to_storage_[compute_collapsed_index(
            canonicalize_tensor_index(
                cpp17::array<size_t, sizeof...(N)>{{N...}},
                make_cpp17_array_from_list<Symm>()),
            make_cpp17_array_from_list<index_list>())];
    return storage_index;
  }

  /// Get the multiplicity of the storage_index
  /// \param storage_index the storage_index of which to get the multiplicity
  SPECTRE_ALWAYS_INLINE static constexpr size_t multiplicity(
      const size_t storage_index) noexcept {
    constexpr auto multiplicity = multiplicity_;
    return gsl::at(multiplicity, storage_index);
  }

  /// Get the array of collapsed index to storage_index
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t,
                                                    number_of_components()>
  collapsed_to_storage() noexcept {
    constexpr auto collapsed_to_storage = collapsed_to_storage_;
    return collapsed_to_storage;
  }

  /// Get the storage_index for the specified collapsed index
  SPECTRE_ALWAYS_INLINE static constexpr int collapsed_to_storage(
      const size_t i) noexcept {
    constexpr auto collapsed_to_storage = collapsed_to_storage_;
    return gsl::at(collapsed_to_storage, i);
  }

  /// Get the array of tensor_index's corresponding to the storage_index's.
  SPECTRE_ALWAYS_INLINE static constexpr const std::array<
      std::array<size_t, sizeof...(Indices) == 0 ? 1 : sizeof...(Indices)>,
      size()>&
  storage_to_tensor_index() noexcept {
    constexpr auto storage_to_tensor = storage_to_tensor_;
    return storage_to_tensor;
  }

  template <typename T>
  SPECTRE_ALWAYS_INLINE static std::string component_name(
      const std::array<T, rank()>& tensor_index,
      const std::array<std::string, rank()>& axis_labels) {
    return ComponentNameImpl<sizeof...(Indices)>::template apply<Structure>(
        tensor_index, axis_labels);
  }
};
}  // namespace Tensor_detail
