// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes for Tensor

#pragma once

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Expressions/Contract.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "ErrorHandling/Error.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename X, typename Symm = Symmetry<>,
          typename IndexList = index_list<>>
class Tensor;
/// \endcond

namespace Tensor_detail {
template <typename T, typename = void>
struct is_tensor : std::false_type {};
template <typename X, typename Symm, typename IndexList>
struct is_tensor<Tensor<X, Symm, IndexList>> : std::true_type {};
}  // namespace Tensor_detail

/*!
 * \ingroup TensorGroup
 * \brief Represents an object with multiple components
 *
 * \details
 * Tensor is a container that represents indexable geometric objects. Each index
 * has its own dimension, valence, and frame and must be either spatial or
 * spacetime. Note that the dimension passed to `SpatialIndex` and
 * `SpacetimeIndex` is always the spatial dimension of the index. Tensors with
 * symmetric indices are stored only once and must be of the same
 * type. A list of available type aliases can be found in the ::tnsr namespace
 * where the adopted conventions are:
 *
 * 1. Upper case for contravariant or upper indices, lower case for covariant or
 * lower indices.
 *
 * 2. `a, b, c, d` are used for spacetime indices while `i, j, k, l` are used
 * for spatial indices.
 *
 * 3. `::Scalar` is not inside the `::tnsr` namespace but is used to represent
 * a scalar with no indices.
 *
 * \example
 * \snippet Test_Tensor.cpp scalar
 * \snippet Test_Tensor.cpp spatial_vector
 * \snippet Test_Tensor.cpp spacetime_vector
 * \snippet Test_Tensor.cpp rank_3_122
 *
 * \tparam X the type held
 * \tparam Symm the ::Symmetry of the indices
 * \tparam IndexList a typelist of \ref SpacetimeIndex "TensorIndexType"'s
 */
template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
class Tensor<X, Symm, IndexList<Indices...>> {
  static_assert(sizeof...(Indices) < 5,
                "If you are sure you need rank 5 or higher Tensor's please "
                "file an issue on GitHub or discuss with a core developer of "
                "SpECTRE.");
  static_assert(
      cpp17::is_same_v<X, std::complex<double>> or
          cpp17::is_same_v<X, double> or
          cpp17::is_same_v<X, ComplexDataVector> or
          cpp17::is_same_v<X, ComplexModalVector> or
          cpp17::is_same_v<X, DataVector> or cpp17::is_same_v<X, ModalVector> or
          is_spin_weighted_of_v<ComplexDataVector, X> or
          is_spin_weighted_of_v<ComplexModalVector, X>,
      "Only a Tensor<std::complex<double>>, Tensor<double>, "
      "Tensor<ComplexDataVector>, Tensor<ComplexModalVector>, "
      "Tensor<DataVector>, Tensor<ModalVector>, "
      "Tensor<SpinWeighted<ComplexDataVector, N>>, "
      "or Tensor<SpinWeighted<ComplexModalVector, N>> is currently "
      "allowed. While other types are technically possible it is not "
      "clear that Tensor is the correct container for them. Please "
      "seek advice on the topic by discussing with the SpECTRE developers.");
  /// The Tensor_detail::Structure for the particular tensor index structure
  ///
  /// Each tensor index structure, e.g. \f$T_{ab}\f$, \f$T_a{}^b\f$ or
  /// \f$T^{ab}\f$ has its own Tensor_detail::TensorStructure that holds
  /// information about how the data is stored, what the multiplicity of the
  /// stored indices are, the number of (independent) components, etc.
  using structure = Tensor_detail::Structure<Symm, Indices...>;

 public:
  /// The type of the sequence that holds the data
  using storage_type =
      std::array<X, Tensor_detail::Structure<Symm, Indices...>::size()>;
  /// The type that is stored by the Tensor
  using type = X;
  /// Typelist of the symmetry of the Tensor
  ///
  /// \details
  /// For a rank-3 tensor symmetric in the last two indices,
  /// \f$T_{a(bc)}\f$, the ::Symmetry is `<2, 1, 1>`. For a non-symmetric rank-2
  /// tensor the ::Symmetry is `<2, 1>`.
  using symmetry = Symm;
  /// Typelist of the \ref SpacetimeIndex "TensorIndexType"'s that the
  /// Tensor has
  using index_list = tmpl::list<Indices...>;
  /// The type of the TensorExpression that would represent this Tensor in a
  /// tensor expression.
  template <typename ArgsList>
  using TE = TensorExpression<Tensor<X, Symm, tmpl::list<Indices...>>, X, Symm,
                              tmpl::list<Indices...>, ArgsList>;

  Tensor() = default;
  ~Tensor() = default;
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  /// \cond HIDDEN_SYMBOLS
  /// Constructor from a TensorExpression.
  ///
  /// \tparam LhsIndices the indices on the LHS of the tensor expression
  /// \tparam T the type of the TensorExpression
  /// \param tensor_expression the tensor expression being evaluated
  template <typename... LhsIndices, typename T,
            Requires<std::is_base_of<Expression, T>::value> = nullptr>
  Tensor(const T& tensor_expression,
         tmpl::list<LhsIndices...> /*meta*/) noexcept {
    static_assert(
        sizeof...(LhsIndices) == sizeof...(Indices),
        "When calling evaluate<...>(...) you must pass the same "
        "number of indices as template parameters as there are free "
        "indices on the resulting tensor. For example, auto F = "
        "evaluate<_a_t, _b_t>(G); if G has 2 free indices and you want "
        "the LHS of the equation to be F_{ab} rather than F_{ba}.");
    for (size_t i = 0; i < size(); ++i) {
      gsl::at(data_, i) = tensor_expression.template get<LhsIndices...>(
          structure::get_canonical_tensor_index(i));
    }
  }
  /// \endcond

  /// Initialize a vector or scalar from an array
  ///
  /// \example
  /// \snippet Test_Tensor.cpp init_vector
  /// \param data the values of the individual components of the Vector
  template <size_t NumberOfIndices = sizeof...(Indices),
            Requires<(NumberOfIndices <= 1)> = nullptr>
  explicit Tensor(storage_type data) noexcept;

  /// Constructor that passes "args" to constructor of X and initializes each
  /// component to be the same
  template <typename... Args,
            Requires<not(cpp17::disjunction_v<std::is_same<
                             Tensor<X, Symm, IndexList<Indices...>>,
                             std::decay_t<Args>>...> and
                         sizeof...(Args) == 1) and
                     cpp17::is_constructible_v<X, Args...>> = nullptr>
  explicit Tensor(Args&&... args) noexcept;

  using value_type = typename storage_type::value_type;
  using reference = typename storage_type::reference;
  using const_reference = typename storage_type::const_reference;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;
  using pointer = typename storage_type::pointer;
  using const_pointer = typename storage_type::const_pointer;
  using reverse_iterator = typename storage_type::reverse_iterator;
  using const_reverse_iterator = typename storage_type::const_reverse_iterator;

  iterator begin() noexcept { return data_.begin(); }
  const_iterator begin() const noexcept { return data_.begin(); }
  const_iterator cbegin() const noexcept { return data_.begin(); }

  iterator end() noexcept { return data_.end(); }
  const_iterator end() const noexcept { return data_.end(); }
  const_iterator cend() const noexcept { return data_.end(); }

  reverse_iterator rbegin() noexcept { return data_.rbegin(); }
  const_reverse_iterator rbegin() const noexcept { return data_.rbegin(); }
  const_reverse_iterator crbegin() const noexcept { return data_.rbegin(); }

  reverse_iterator rend() noexcept { return data_.rend(); }
  const_reverse_iterator rend() const noexcept { return data_.rend(); }
  const_reverse_iterator crend() const noexcept { return data_.rend(); }

  // @{
  /// Get data entry using an array representing a tensor index
  ///
  /// \details
  /// Let \f$T_{abc}\f$ be a Tensor.
  /// Then `get({{0, 2, 1}})` returns the \f$T_{0 2 1}\f$ component.
  /// \param tensor_index the index at which to get the data
  template <typename T>
  SPECTRE_ALWAYS_INLINE constexpr reference get(
      const std::array<T, sizeof...(Indices)>& tensor_index) noexcept {
    return gsl::at(data_, structure::get_storage_index(tensor_index));
  }
  template <typename T>
  SPECTRE_ALWAYS_INLINE constexpr const_reference get(
      const std::array<T, sizeof...(Indices)>& tensor_index) const noexcept {
    return gsl::at(data_, structure::get_storage_index(tensor_index));
  }
  // @}
  // @{
  /// Get data entry using a list of integers representing a tensor index
  ///
  /// \details
  /// Let \f$T_{abc}\f$ be a Tensor.
  /// Then `get(0, 2, 1)` returns the \f$T_{0 2 1}\f$ component.
  /// \param n the index at which to get the data
  template <typename... N>
  constexpr reference get(N... n) noexcept {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    return gsl::at(data_, structure::get_storage_index(n...));
  }
  template <typename... N>
  constexpr const_reference get(N... n) const noexcept {
    static_assert(
        sizeof...(Indices) == sizeof...(N),
        "the number of tensor indices specified must match the rank of "
        "the tensor");
    return gsl::at(data_, structure::get_storage_index(n...));
  }
  // @}

  // @{
  /// Retrieve the index `N...` by computing the storage index at compile time
  // clang-tidy: redundant declaration (bug in clang-tidy)
  template <int... N, typename... Args>
  friend SPECTRE_ALWAYS_INLINE constexpr typename Tensor<Args...>::reference
  get(Tensor<Args...>& t) noexcept;  // NOLINT
  // clang-tidy: redundant declaration (bug in clang-tidy)
  template <int... N, typename... Args>
  friend SPECTRE_ALWAYS_INLINE constexpr
      typename Tensor<Args...>::const_reference
      get(const Tensor<Args...>& t) noexcept;  // NOLINT
  // @}

  // @{
  /// Retrieve a TensorExpression object with the index structure passed in
  template <typename... N,
            Requires<cpp17::conjunction_v<tt::is_tensor_index<N>...> and
                     tmpl::is_set<N...>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE constexpr TE<tmpl::list<N...>> operator()(
      N... /*meta*/) const noexcept {
    return TE<tmpl::list<N...>>(*this);
  }

  template <typename... N,
            Requires<cpp17::conjunction_v<tt::is_tensor_index<N>...> and
                     not tmpl::is_set<N...>::value> = nullptr>
  SPECTRE_ALWAYS_INLINE constexpr auto operator()(N... /*meta*/) const noexcept
      -> decltype(
          TensorExpressions::fully_contracted<
              TE, replace_indices<tmpl::list<N...>, repeated<tmpl::list<N...>>>,
              tmpl::int32_t<0>,
              tmpl::size<repeated<tmpl::list<N...>>>>::apply(*this)) {
    using args_list = tmpl::list<N...>;
    using repeated_args_list = repeated<args_list>;
    return TensorExpressions::fully_contracted<
        TE, replace_indices<args_list, repeated_args_list>, tmpl::int32_t<0>,
        tmpl::size<repeated_args_list>>::apply(*this);
  }
  // @}

  // @{
  /// Return i'th component of storage vector
  constexpr reference operator[](const size_t storage_index) noexcept {
    return gsl::at(data_, storage_index);
  }
  constexpr const_reference operator[](const size_t storage_index) const
      noexcept {
    return gsl::at(data_, storage_index);
  }
  // @}

  /// Return the number of independent components of the Tensor
  ///
  /// \details
  /// Returns the number of independent components of the Tensor taking into
  /// account symmetries. For example, let \f$T_{ab}\f$ be a n-dimensional
  /// rank-2 symmetric tensor, then the number of independent components is
  /// \f$n(n+1)/2\f$.
  SPECTRE_ALWAYS_INLINE static constexpr size_t size() noexcept {
    return structure::size();
  }

  /// Returns the rank of the Tensor
  ///
  /// \details
  /// The rank of a tensor is the number of indices it has. For example, the
  /// tensor \f$v^a\f$ is rank-1, the tensor \f$\phi\f$ is rank-0, and the
  /// tensor \f$T_{abc}\f$ is rank-3.
  SPECTRE_ALWAYS_INLINE static constexpr size_t rank() noexcept {
    return sizeof...(Indices);
  }

  // @{
  /// Given an iterator or storage index, get the canonical tensor index.
  /// For scalars this is defined to be std::array<int, 1>{{0}}
  SPECTRE_ALWAYS_INLINE constexpr std::array<size_t, sizeof...(Indices)>
  get_tensor_index(const const_iterator& iter) const noexcept {
    return structure::get_canonical_tensor_index(
        static_cast<size_t>(iter - begin()));
  }
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  get_tensor_index(const size_t storage_index) noexcept {
    return structure::get_canonical_tensor_index(storage_index);
  }
  // @}

  // @{
  /// Get the storage index of the tensor index. Should only be used when
  /// optimizing code in which computing the storage index is a bottleneck.
  template <typename... N>
  SPECTRE_ALWAYS_INLINE static constexpr size_t get_storage_index(
      const N... args) noexcept {
    return structure::get_storage_index(args...);
  }
  template <typename I>
  SPECTRE_ALWAYS_INLINE static constexpr size_t get_storage_index(
      const std::array<I, sizeof...(Indices)>& tensor_index) noexcept {
    return structure::get_storage_index(tensor_index);
  }
  // @}

  // @{
  /// Given an iterator or storage index, get the multiplicity of an index
  ///
  /// \see TensorMetafunctions::compute_multiplicity
  SPECTRE_ALWAYS_INLINE constexpr size_t multiplicity(
      const iterator& iter) const noexcept {
    return structure::multiplicity(static_cast<size_t>(iter - begin()));
  }
  SPECTRE_ALWAYS_INLINE static constexpr size_t multiplicity(
      const size_t storage_index) noexcept {
    return structure::multiplicity(storage_index);
  }
  // @}

  // @{
  /// Get dimensionality of i'th tensor index
  ///
  /// \snippet Test_Tensor.cpp index_dim
  /// \see ::index_dim
  SPECTRE_ALWAYS_INLINE static constexpr size_t index_dim(
      const size_t i) noexcept {
    static_assert(sizeof...(Indices),
                  "A scalar does not have any indices from which you can "
                  "retrieve the dimensionality.");
    return gsl::at(structure::dims(), i);
  }
  // @}

  //@{
  /// Return an array corresponding to the ::Symmetry of the Tensor
  SPECTRE_ALWAYS_INLINE static constexpr std::array<int, sizeof...(Indices)>
  symmetries() noexcept {
    return structure::symmetries();
  }
  //@}

  //@{
  /// Return array of the ::IndexType's (spatial or spacetime)
  SPECTRE_ALWAYS_INLINE static constexpr std::array<IndexType,
                                                    sizeof...(Indices)>
  index_types() noexcept {
    return structure::index_types();
  }
  //@}

  //@{
  /// Return array of dimensionality of each index
  ///
  /// \snippet Test_Tensor.cpp index_dim
  /// \see index_dim ::index_dim
  SPECTRE_ALWAYS_INLINE static constexpr std::array<size_t, sizeof...(Indices)>
  index_dims() noexcept {
    return structure::dims();
  }
  //@}

  //@{
  /// Return array of the valence of each index (::UpLo)
  SPECTRE_ALWAYS_INLINE static constexpr std::array<UpLo, sizeof...(Indices)>
  index_valences() noexcept {
    return structure::index_valences();
  }
  //@}

  //@{
  /// Returns std::tuple of the ::Frame of each index
  SPECTRE_ALWAYS_INLINE static constexpr auto index_frames() noexcept {
    return Tensor_detail::Structure<Symm, Indices...>::index_frames();
  }
  //@}

  //@{
  /// Given a tensor index, get the canonical label associated with the
  /// canonical \ref SpacetimeIndex "TensorIndexType"
  template <typename T = int>
  static std::string component_name(
      const std::array<T, rank()>& tensor_index = std::array<T, rank()>{},
      const std::array<std::string, rank()>& axis_labels =
          make_array<rank()>(std::string(""))) noexcept {
    return structure::component_name(tensor_index, axis_labels);
  }
  //@}

  /// Copy tensor data into an `std::vector<X>` along with the
  /// component names into a `std::vector<std::string>`
  /// \requires `std::is_same<X, DataVector>::%value` is true
  std::pair<std::vector<std::string>, std::vector<X>> get_vector_of_data() const
      noexcept;

  /// \cond HIDDEN_SYMBOLS
  /// Serialization function used by Charm++
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | data_;
  }
  /// \endcond

 private:
  // clang-tidy: redundant declaration
  /// \cond
  template <int I, class... Ts>
  friend SPECTRE_ALWAYS_INLINE constexpr size_t index_dim(  // NOLINT
      const Tensor<Ts...>& /*t*/) noexcept;
  /// \endcond

  storage_type data_;
};

// ================================================================
// Template Definitions - Variadic templates must be in header
// ================================================================

template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
template <size_t NumberOfIndices, Requires<(NumberOfIndices <= 1)>>
Tensor<X, Symm, IndexList<Indices...>>::Tensor(storage_type data) noexcept
    : data_(std::move(data)) {}

// The cpp17::disjunction is used to prevent the compiler from matching this
// function when it should select the move constructor.
template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
template <typename... Args,
          Requires<not(cpp17::disjunction_v<
                           std::is_same<Tensor<X, Symm, IndexList<Indices...>>,
                                        std::decay_t<Args>>...> and
                       sizeof...(Args) == 1) and
                   cpp17::is_constructible_v<X, Args...>>>
Tensor<X, Symm, IndexList<Indices...>>::Tensor(Args&&... args) noexcept
    : data_(make_array<size(), X>(std::forward<Args>(args)...)) {}

template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
std::pair<std::vector<std::string>, std::vector<X>>
Tensor<X, Symm, IndexList<Indices...>>::get_vector_of_data() const noexcept {
  std::vector<value_type> serialized_tensor(size());
  std::vector<std::string> component_names(size());
  for (size_t i = 0; i < data_.size(); ++i) {
    component_names[i] = component_name(get_tensor_index(i));
    serialized_tensor[i] = gsl::at(data_, i);
  }
  return std::make_pair(component_names, serialized_tensor);
}

template <int... N, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr typename Tensor<Args...>::reference get(
    Tensor<Args...>& t) noexcept {
  static_assert(Tensor<Args...>::rank() == sizeof...(N),
                "the number of tensor indices specified must match the rank "
                "of the tensor");
  return gsl::at(
      t.data_, Tensor<Args...>::structure::template get_storage_index<N...>());
}

template <int... N, typename... Args>
SPECTRE_ALWAYS_INLINE constexpr typename Tensor<Args...>::const_reference get(
    const Tensor<Args...>& t) noexcept {
  static_assert(Tensor<Args...>::rank() == sizeof...(N),
                "the number of tensor indices specified must match the rank "
                "of the tensor");
  return gsl::at(
      t.data_, Tensor<Args...>::structure::template get_storage_index<N...>());
}

template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
bool operator==(const Tensor<X, Symm, IndexList<Indices...>>& lhs,
                const Tensor<X, Symm, IndexList<Indices...>>& rhs) noexcept {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}
template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
bool operator!=(const Tensor<X, Symm, IndexList<Indices...>>& lhs,
                const Tensor<X, Symm, IndexList<Indices...>>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \ingroup TensorGroup
/// Get dimensionality of i'th tensor index
///
/// \snippet Test_Tensor.cpp index_dim
template <int I, class... Ts>
SPECTRE_ALWAYS_INLINE constexpr size_t index_dim(
    const Tensor<Ts...>& /*t*/) noexcept {
  return Tensor<Ts...>::structure::template dim<I>();
}

// We place the stream operators in the header file so they do not need to be
// explicitly instantiated.
template <typename X, typename Symm, template <typename...> class IndexList,
          typename... Indices>
std::ostream& operator<<(
    std::ostream& os,
    const Tensor<X, Symm, IndexList<Indices...>>& x) noexcept {
  static_assert(tt::is_streamable_v<decltype(os), X>,
                "operator<< is not defined for the type you are trying to "
                "stream in Tensor");
  for (size_t i = 0; i < x.size() - 1; ++i) {
    os << "T" << x.get_tensor_index(i) << "=" << x[i]
       << "\n";
  }
  size_t i = x.size() - 1;
  os << "T" << x.get_tensor_index(i) << "=" << x[i];
  return os;
}

namespace MakeWithValueImpls {
template <typename... Structure>
struct MakeWithValueImpl<Tensor<DataVector, Structure...>, DataVector> {
  /// \brief Returns a Tensor whose DataVectors are the same size as `input`,
  /// with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE Tensor<DataVector, Structure...> apply(
      const DataVector& input, const double value) noexcept {
    return Tensor<DataVector, Structure...>(input.size(), value);
  }
};

template <typename... Structure>
struct MakeWithValueImpl<DataVector, Tensor<DataVector, Structure...>> {
  /// \brief Returns a DataVector with the same size as the DataVectors of
  /// `input`, with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE DataVector
  apply(const Tensor<DataVector, Structure...>& input,
        const double value) noexcept {
    return DataVector(input.begin()->size(), value);
  }
};

template <typename... StructureOut, typename... StructureIn>
struct MakeWithValueImpl<Tensor<DataVector, StructureOut...>,
                         Tensor<DataVector, StructureIn...>> {
  /// \brief Returns a Tensor whose DataVectors are the same size as the
  /// DataVectors of `input`, with each element equal to `value`.
  static SPECTRE_ALWAYS_INLINE Tensor<DataVector, StructureOut...> apply(
      const Tensor<DataVector, StructureIn...>& input,
      const double value) noexcept {
    return Tensor<DataVector, StructureOut...>(input.begin()->size(), value);
  }
};

template <typename... Structure>
struct MakeWithValueImpl<Tensor<double, Structure...>, double> {
  /// \brief Returns a Tensor whose elements are set equal to `value` (`input`
  /// is ignored).
  static SPECTRE_ALWAYS_INLINE Tensor<double, Structure...> apply(
      const double& /*input*/, const double value) noexcept {
    return Tensor<double, Structure...>(value);
  }
};

template <typename... Structure>
struct MakeWithValueImpl<double, Tensor<double, Structure...>> {
  /// \brief Returns a double initialized to `value` (`input` is ignored)
  static SPECTRE_ALWAYS_INLINE double apply(
      const Tensor<double, Structure...>& /*input*/,
      const double value) noexcept {
    return value;
  }
};

template <typename... StructureOut, typename... StructureIn>
struct MakeWithValueImpl<Tensor<double, StructureOut...>,
                         Tensor<double, StructureIn...>> {
  /// \brief Returns a Tensor whose elements are set equal to `value` (`input`
  /// is ignored).
  static SPECTRE_ALWAYS_INLINE Tensor<double, StructureOut...> apply(
      const Tensor<double, StructureIn...>& /*input*/,
      const double value) noexcept {
    return Tensor<double, StructureOut...>(value);
  }
};
}  // namespace MakeWithValueImpls
