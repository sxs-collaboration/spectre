// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Constraints.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief Tags for the scalar tensor system.
 */
namespace ScalarTensor {
namespace Tags {
/*!
 * \brief Represents the trace-reversed stress-energy tensor of the scalar
 * field.
 */
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct TraceReversedStressEnergy : db::SimpleTag {
  using type = tnsr::aa<DataType, Dim, Fr>;
};

/*!
 * \brief Tag holding the source term of the scalar equation.
 *
 * \details This tag hold the source term \f$ \mathcal{S} \f$,
 * entering a wave equation of the form
 * \f[
 *   \Box \Psi = \mathcal{S} ~.
 * \f]
 */
struct ScalarSource : db::SimpleTag {
  using type = Scalar<DataVector>;
};

}  // namespace Tags

namespace OptionTags {
/*!
 * \brief Scalar mass parameter.
 */
struct ScalarMass {
  static std::string name() { return "ScalarMass"; }
  using type = double;
  static constexpr Options::String help{
      "Mass of the scalar field in code units"};
};
}  // namespace OptionTags

namespace Tags {
struct ScalarMass : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::ScalarMass>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double mass_psi) { return mass_psi; }
};

/*!
 * \brief Prefix tag to avoid ambiguities when observing variables with the same
 * name in both parent systems.
 * \note Since we also add compute tags for these quantities, we do not make
 * this a derived class of `Tag`. Otherwise, we would have tags with repeated
 * base tags in the `ObservationBox`.
 */
template <typename Tag>
struct Csw : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Compute tag to retrieve the values of CurvedScalarWave variables
 * with the same name as in the ::gh systems.
 * \note Since we also add compute tags for these quantities, we do not make
 * this a derived class of `Tag`. Otherwise, we would have tags with repeated
 * base tags in the `ObservationBox`.
 */
template <typename Tag>
struct CswCompute : Csw<Tag>, db::ComputeTag {
  using argument_tags = tmpl::list<Tag>;
  using base = Csw<Tag>;
  using return_type = typename base::type;
  static constexpr void function(const gsl::not_null<return_type*> result,
                                 const return_type& csw_var) {
    for (size_t i = 0; i < csw_var.size(); ++i) {
      make_const_view(make_not_null(&std::as_const((*result)[i])), csw_var[i],
                      0, csw_var[i].size());
    }
  }
};

/*!
 * \brief Computes the scalar-wave one-index constraint.
 * \details The one-index constraint is assigned to a wrapped tag to avoid
 * clashes with the ::gh constraints during observation.
 * \note We do not use ScalarTensor::Tags::CswCompute to retrieve the
 * CurvedScalarWave constraints since we do not want to add the bare compute
 * tags (::CurvedScalarWave::Tags::OneIndexConstraintCompute and
 * ::CurvedScalarWave::Tags::TwoIndexConstraintCompute) directly in the
 * ObservationBox, since that will make them observable and would lead to a
 * clash with the ::gh constraint tags.
 */
template <size_t Dim>
struct CswOneIndexConstraintCompute
    : Csw<CurvedScalarWave::Tags::OneIndexConstraint<Dim>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<Dim>,
                               Frame::Inertial>,
                 CurvedScalarWave::Tags::Phi<Dim>>;
  using return_type = tnsr::i<DataVector, Dim>;
  static constexpr void (*function)(const gsl::not_null<return_type*> result,
                                    const tnsr::i<DataVector, Dim>&,
                                    const tnsr::i<DataVector, Dim>&) =
      &CurvedScalarWave::one_index_constraint<Dim>;
  using base = Csw<CurvedScalarWave::Tags::OneIndexConstraint<Dim>>;
};

/*!
 * \brief Computes the scalar-wave two-index constraint.
 * \details The two-index constraint is assigned to a wrapped tag to avoid
 * clashes with the ::gh constraints during observation.
 * \note We do not use ScalarTensor::Tags::CswCompute to retrieve the
 * CurvedScalarWave constraints since we do not want to add the bare compute
 * tags (::CurvedScalarWave::Tags::OneIndexConstraintCompute and
 * ::CurvedScalarWave::Tags::TwoIndexConstraintCompute) directly in the
 * ObservationBox, since that will make them observable and would lead to a
 * clash with the ::gh constraint tags.
 */
template <size_t Dim>
struct CswTwoIndexConstraintCompute
    : Csw<CurvedScalarWave::Tags::TwoIndexConstraint<Dim>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<::Tags::deriv<CurvedScalarWave::Tags::Phi<Dim>,
                               tmpl::size_t<Dim>, Frame::Inertial>>;
  using return_type = tnsr::ij<DataVector, Dim>;
  static constexpr void (*function)(const gsl::not_null<return_type*> result,
                                    const tnsr::ij<DataVector, Dim>&) =
      &CurvedScalarWave::two_index_constraint<Dim>;
  using base = Csw<CurvedScalarWave::Tags::TwoIndexConstraint<Dim>>;
};

}  // namespace Tags

}  // namespace ScalarTensor
