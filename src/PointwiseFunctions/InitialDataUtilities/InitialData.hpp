// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace evolution {
/*!
 * \brief Namespace for things related to initial data used for evolution
 * systems.
 */
namespace initial_data {
/*!
 * \brief The abstract base class for initial data of evolution systems. All
 * analytic solutions and analytic data must virtually inherit from this class.
 */
class InitialData : public PUP::able {
 protected:
  InitialData() = default;

 public:
  ~InitialData() override = default;

  virtual auto get_clone() const -> std::unique_ptr<InitialData> = 0;

  /*!
   * \brief Unwrap wrapper types (e.g. `WithNoise`, which wraps analytic
   * initial data and adds random noise) to reach the underlying analytic
   * solution or data.
   *
   * Most classes return `*this`. `WithNoise` overrides this to return its
   * inner solution, allowing dispatch sites to transparently look through
   * the wrapper without knowing about it explicitly.
   */
  virtual const InitialData& unwrap() const { return *this; }

  /// \cond
  explicit InitialData(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(InitialData);
  /// \endcond
};
}  // namespace initial_data
}  // namespace evolution
