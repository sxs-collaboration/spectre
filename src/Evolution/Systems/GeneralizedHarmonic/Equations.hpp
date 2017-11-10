// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template GeneralizedHarmonicEquations.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TagsDeclarations.hpp"

class DataVector;

namespace GeneralizedHarmonic {
template <size_t Dim>
struct ComputeDuDt {
 public:
  using return_tags = typelist<Tags::dt<SpacetimeMetric<Dim>>,
                               Tags::dt<Pi<Dim>>, Tags::dt<Phi<Dim>>>;
  using argument_tags = typelist<
      SpacetimeMetric<Dim>, Pi<Dim>, Phi<Dim>,
      Tags::d<SpacetimeMetric<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::d<Pi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::d<Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>, ConstraintGamma0,
      ConstraintGamma1, ConstraintGamma2, GaugeH<Dim>,
      SpacetimeDerivGaugeH<Dim>, Lapse<Dim>, Shift<Dim>,
      InverseSpatialMetric<Dim>, InverseSpacetimeMetric<Dim>,
      TraceSpacetimeChristoffelFirstKind<Dim>,
      SpacetimeChristoffelFirstKind<Dim>, SpacetimeChristoffelSecondKind<Dim>,
      SpacetimeNormalVector<Dim>, SpacetimeNormalOneForm<Dim>>;

  /*!
   * /brief Compute the RHS of the Generalized Harmonic formulation of
   * Einstein's equations.
   *
   * /details For the full form of the equations see "A New Generalized Harmonic
   * Evolution System" by Lindblom et. al, arxiv.org/abs/gr-qc/0512093.
   */
  static void apply(
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_spacetime_metric,
      gsl::not_null<tnsr::aa<DataVector, Dim>*> dt_pi,
      gsl::not_null<tnsr::iaa<DataVector, Dim>*> dt_phi,
      const tnsr::aa<DataVector, Dim>& spacetime_metric,
      const tnsr::aa<DataVector, Dim>& pi,
      const tnsr::iaa<DataVector, Dim>& phi,
      const tnsr::iaa<DataVector, Dim>& d_spacetime_metric,
      const tnsr::iaa<DataVector, Dim>& d_pi,
      const tnsr::ijaa<DataVector, Dim>& d_phi,
      const Scalar<DataVector>& gamma0, const Scalar<DataVector>& gamma1,
      const Scalar<DataVector>& gamma2,
      const tnsr::a<DataVector, Dim>& gauge_function,
      const tnsr::ab<DataVector, Dim>& spacetime_deriv_gauge_function,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
      const tnsr::II<DataVector, Dim>& inverse_spatial_metric,
      const tnsr::AA<DataVector, Dim>& inverse_spacetime_metric,
      const tnsr::a<DataVector, Dim>& trace_christoffel,
      const tnsr::abb<DataVector, Dim>& christoffel_first_kind,
      const tnsr::Abb<DataVector, Dim>& christoffel_second_kind,
      const tnsr::A<DataVector, Dim>& normal_spacetime_vector,
      const tnsr::a<DataVector, Dim>& normal_spacetime_one_form);
};
}  // namespace GeneralizedHarmonic
