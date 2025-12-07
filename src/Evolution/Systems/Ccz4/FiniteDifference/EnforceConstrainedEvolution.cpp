// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/EnforceConstrainedEvolution.hpp"

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {

void EnforceConstrainedEvolution::apply(
    const gsl::not_null<tnsr::ii<DataVector, dim>*> conformal_spatial_metric,
    const gsl::not_null<tnsr::ii<DataVector, dim>*> a_tilde,
    const bool constrained_evolution) {
  if (constrained_evolution) {
    // Allocate shared storage for temporaries in one block using existing tags
    Variables<tmpl::list<Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
                         Ccz4::Tags::TraceATilde<DataVector>,
                         Ccz4::Tags::InverseConformalMetric<DataVector, dim>>>
        temporaries((conformal_spatial_metric->get(0, 0)).size());

    auto& det_conformal_spatial_metric =
        get<Ccz4::Tags::DetConformalSpatialMetric<DataVector>>(temporaries);
    auto& trace_a_tilde = get<Ccz4::Tags::TraceATilde<DataVector>>(temporaries);
    auto& inv_conformal_spatial_metric =
        get<Ccz4::Tags::InverseConformalMetric<DataVector, dim>>(temporaries);

    determinant(make_not_null(&det_conformal_spatial_metric),
                *conformal_spatial_metric);
    ASSERT(min(get(det_conformal_spatial_metric)) > 0.0,
           "The determinant of the conformal spatial metric is non-positive: "
               << get(det_conformal_spatial_metric));
    get(det_conformal_spatial_metric) =
        pow(get(det_conformal_spatial_metric), -1.0 / 3.0);
    ::tenex::update<ti::i, ti::j>(
        conformal_spatial_metric,
        det_conformal_spatial_metric() *
            (*conformal_spatial_metric)(ti::i, ti::j));

    determinant_and_inverse(make_not_null(&det_conformal_spatial_metric),
                            make_not_null(&inv_conformal_spatial_metric),
                            *conformal_spatial_metric);
    ::tenex::evaluate(make_not_null(&trace_a_tilde),
                      (inv_conformal_spatial_metric)(ti::I, ti::J) *
                          (*a_tilde)(ti::i, ti::j));
    ::tenex::update<ti::i, ti::j>(
        a_tilde,
        (*a_tilde)(ti::i, ti::j) -
            trace_a_tilde() * (*conformal_spatial_metric)(ti::i, ti::j) / 3.);
  }
}

}  // namespace Ccz4::fd
