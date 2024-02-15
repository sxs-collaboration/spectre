// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

/*!
 * \brief Simple struct that changes a class from being factory creatable, to
 * non-factory creatable.
 *
 * It does this by defining and setting a static constexpr bool
 * `factory_creatable = false`.
 *
 * \tparam FactoryCreatableClass Any class that is already factory creatable.
 */
template <typename FactoryCreatableClass>
struct NonFactoryCreatableWrapper : public FactoryCreatableClass {
  static constexpr bool factory_creatable = false;
  using FactoryCreatableClass::FactoryCreatableClass;

  using factory_creatable_class = FactoryCreatableClass;

  /// \cond
  explicit NonFactoryCreatableWrapper(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NonFactoryCreatableWrapper);  // NOLINT
  /// \endcond
};

/// \cond
// NOLINTBEGIN
template <typename FactoryCreatableClass>
PUP::able::PUP_ID NonFactoryCreatableWrapper<FactoryCreatableClass>::my_PUP_ID =
    0;
// NOLINTEND
/// \endcond
