// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup AnalyticDataGroup
 * \brief Holds analytic profiles for the scalar field in scalar-tensor
 * theories.
 */
namespace ScalarTensor::AnalyticData::ScalarField {

/*!
 * \brief Base struct representing the analytic profile for the scalar field in
 * scalar-tensor theories.
 */
template <size_t Dim>
struct AnalyticData : public virtual PUP::able {
 public:
  using scalar_field_tags =
      tmpl::list<CurvedScalarWave::Tags::Psi,
                 CurvedScalarWave::Tags::Phi<Dim, Frame::Inertial>>;

 protected:
  AnalyticData() = default;

 public:
  ~AnalyticData() override = default;

  /// \cond
  explicit AnalyticData(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(AnalyticData);
  /// \endcond

  virtual std::unique_ptr<AnalyticData> get_clone() const = 0;
};

}  // namespace ScalarTensor::AnalyticData::ScalarField
