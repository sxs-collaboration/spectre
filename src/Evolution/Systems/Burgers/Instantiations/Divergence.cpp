// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"

template Variables<tmpl::list<
    Tags::div<Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>>
divergence(
    const Variables<tmpl::list<
        Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>& F,
    const Mesh<1>& mesh,
    const InverseJacobian<DataVector, 1, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian);
