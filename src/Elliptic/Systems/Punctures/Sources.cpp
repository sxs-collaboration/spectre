// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Punctures/Sources.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Punctures {

void add_sources(const gsl::not_null<Scalar<DataVector>*> puncture_equation,
                 const Scalar<DataVector>& alpha,
                 const Scalar<DataVector>& beta,
                 const Scalar<DataVector>& field) {
  get(*puncture_equation) +=
      get(beta) / pow<7>(get(alpha) * (get(field) + 1.) + 1.);
}

void add_linearized_sources(
    const gsl::not_null<Scalar<DataVector>*> linearized_puncture_equation,
    const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
    const Scalar<DataVector>& field,
    const Scalar<DataVector>& field_correction) {
  get(*linearized_puncture_equation) -=
      7. * get(alpha) * get(beta) /
      pow<8>(get(alpha) * (get(field) + 1.) + 1.) * get(field_correction);
}

void Sources::apply(const gsl::not_null<Scalar<DataVector>*> puncture_equation,
                    const Scalar<DataVector>& alpha,
                    const Scalar<DataVector>& beta,
                    const Scalar<DataVector>& field,
                    const tnsr::I<DataVector, 3>& /*field_flux*/) {
  add_sources(puncture_equation, alpha, beta, field);
}

void Sources::apply(
    const gsl::not_null<
        tnsr::i<DataVector, 3>*> /*equation_for_field_gradient*/,
    const Scalar<DataVector>& /*alpha*/, const Scalar<DataVector>& /*beta*/,
    const Scalar<DataVector>& /*field*/) {}

void LinearizedSources::apply(
    const gsl::not_null<Scalar<DataVector>*> linearized_puncture_equation,
    const Scalar<DataVector>& alpha, const Scalar<DataVector>& beta,
    const Scalar<DataVector>& field, const Scalar<DataVector>& field_correction,
    const tnsr::I<DataVector, 3>& /*field_flux_correction*/) {
  add_linearized_sources(linearized_puncture_equation, alpha, beta, field,
                         field_correction);
}

void LinearizedSources::apply(
    const gsl::not_null<
        tnsr::i<DataVector, 3>*> /*equation_for_field_gradient_correction*/,
    const Scalar<DataVector>& /*alpha*/, const Scalar<DataVector>& /*beta*/,
    const Scalar<DataVector>& /*field*/,
    const Scalar<DataVector>& /*field_correction*/) {}

}  // namespace Punctures
