// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::Object abstract base class

#pragma once

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Abstract base class representing an object in an HDF5 file
 */
class Object {
 public:
  /// \cond HIDDEN_SYMBOLS
  Object() = default;
  Object(const Object& /*rhs*/) = delete;
  Object& operator=(const Object& /*rhs*/) = delete;
  Object(Object&& /*rhs*/) noexcept = delete;             // NOLINT
  Object& operator=(Object&& /*rhs*/) noexcept = delete;  // NOLINT
  virtual ~Object() = default;
  /// \endcond
};
}  // namespace h5
