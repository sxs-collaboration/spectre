// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/ScalarField/AnalyticData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarTensor::AnalyticData::ScalarField {

/*!
 * \brief Vanishing profile for the scalar field.
 */
template <size_t Dim>
class Zero : public ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim> {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "The trivial solution, useful as initial guess."};

  using scalar_field_tags =
      typename ScalarTensor::AnalyticData::ScalarField::AnalyticData<
          Dim>::scalar_field_tags;

  Zero() = default;
  Zero(const Zero&) = default;
  Zero& operator=(const Zero&) = default;
  Zero(Zero&&) = default;
  Zero& operator=(Zero&&) = default;
  ~Zero() override = default;
  std::unique_ptr<ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>>
  get_clone() const override;

  /// \cond
  explicit Zero(CkMigrateMessage* m)
      : ScalarTensor::AnalyticData::ScalarField::AnalyticData<Dim>(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Zero);  // NOLINT
  /// \endcond

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(
        tmpl::size<tmpl::list_difference<tmpl::list<RequestedTags...>,
                                         scalar_field_tags>>::value == 0,
        "The requested tag is not supported");
    return {make_with_value<typename RequestedTags::type>(x, 0.)...};
  }

  template <size_t SpatialDim>
  friend bool operator==(const Zero<SpatialDim>& /*lhs*/,
                         const Zero<SpatialDim>& /*rhs*/);
};

template <size_t SpatialDim>
bool operator!=(const Zero<SpatialDim>& lhs, const Zero<SpatialDim>& rhs);

}  // namespace ScalarTensor::AnalyticData::ScalarField
