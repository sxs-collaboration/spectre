// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

namespace TestHelpers::Tags {
/// [simple_variables_tag]
template <typename VectorType = DataVector>
struct Vector : ::db::SimpleTag {
  using type = tnsr::I<VectorType, 3>;
};
/// [simple_variables_tag]

template <typename VectorType = DataVector>
struct OneForm : ::db::SimpleTag {
  using type = tnsr::i<VectorType, 3>;
};

template <typename VectorType = DataVector>
struct Scalar : ::db::SimpleTag {
  using type = ::Scalar<VectorType>;
};

template <typename VectorType = DataVector>
struct Scalar2 : ::db::SimpleTag {
  using type = ::Scalar<VectorType>;
};

template <typename VectorType = DataVector>
struct DerivOfVector : ::db::SimpleTag {
  using type = tnsr::iJ<VectorType, 3>;
};

/// [prefix_variables_tag]
template <class Tag>
struct Prefix0 : ::db::PrefixTag, ::db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
/// [prefix_variables_tag]

template <class Tag>
struct Prefix1 : ::db::PrefixTag, ::db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <class Tag>
struct Prefix2 : ::db::PrefixTag, ::db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

template <class Tag>
struct Prefix3 : ::db::PrefixTag, ::db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
}  // namespace TestHelpers::Tags
