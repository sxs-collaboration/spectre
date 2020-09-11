\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Observers Infrastructure {#observers_infrastructure_dev_guide}

The observers infrastructure works with two parallel components: a group and a
nodegroup. We have two types of observations: `Reduction` and `Volume` (see the
enum `observers::TypeOfObservation`). `Reduction` data is anything that is
written once per time/integral identifier per simulation. Some examples of
reduction data are integrals or L2 norms over the entire domain, integrals or L2
norms over part of the domain, and integrals over lower-dimensional surfaces
such as apparent horizons or slices through the domain. Volume data is anything
that has physical extent, such as any of the evolved variables (or derived
quantities thereof) across all or part of the domain, or quantities on
lower-dimensional surfaces in the domain (e.g. the rest mass density in the
xy-plane). Reduction and volume data both use the group and nodegroup for
actually getting the data to disk, but do so in a slightly different manner.

### Reduction Data

Reduction data requires combining information from many or all cores of a
supercomputer to get a single value. Reductions are tagged by some temporal
value, which for hyperbolic systems is the time and for elliptic systems some
combination of linear and non-linear iteration count. The reduction data is
stored in an object of type `Parallel::ReductionData`, which takes as template
parameters a series of `Parallel::ReductionDatum`. A `Parallel::ReductionDatum`
takes as template parameters the type of the data and operators that define how
data from the different cores are to be combined to a single value. See the
paragraphs below for more detail, and the documentation of
`Parallel::ReductionDatum` for examples.

At the start of a simulation, every component and event that wants to perform a
reduction for observation, or will be part of a reduction observation, must
register with the `observers::Observer` component. The `observers::Observer` is
a group, which means there is one per core. The registration is used so that the
`Observer` knows once all data for a specific reduction (both in time and by
name/ID) has been contributed. Reduction data is combined on each core as it is
contributed by using the binary operator from `Parallel::ReductionDatum`'s
second template parameter. Once all the data is collected on the core, it is
copied to the local `observers::ObserverWriter` nodegroup, which keeps track of
how many of the cores on the node will be contributing to a specific
observation, and again combines all the data as it is being contributed. Once
all the node's data is collected to the nodegroup, the data is sent to node `0`
which combines the reduction data as it arrives using the binary operator from
`Parallel::ReductionDatum`'s second template parameter. Using node `0` for
collecting the final reduction data is an arbitrary choice, but we are always
guaranteed to have a node `0`.

Once all the reductions are received on node `0`, the `ObserverWriter` invokes
the `InvokeFinal` (third) template parameter on each `Parallel::ReductionDatum`
(this is the n-ary) in order to finalize the data before writing. This is used,
for example, for dividing by the total number of grid points in an L1 or L2
norm. The reduction data is then written to an HDF5 file whose name is set in
the input file using the option
`observers::Tags::ReductionFileName`. Specifically, the data is written into an
`h5::Dat` subfile since, along with the data, the subfile name must be passed
through the reductions.

The actions used for registering reductions are
`observers::Actions::RegisterEventsWithObservers`,
`observers::Actions::RegisterSingletonWithObserverWriter`, and
`observers::Actions::RegisterWithObservers`. There is a separate `Registration`
phase at the beginning of all simulations where everything must register with
the observers. The action `observers::Actions::ContributeReductionData` is used
to send data to the `observers::Observer` component in the case where there is a
reduction done across an array or subset of an array. If a singleton parallel
component needs to write data directly to disk it should use the
`observers::ThreadedActions::WriteReductionData` action called on the zeroth
element of the `observers::ObserverWriter` component.

### Volume Data

Volume data is vaguely defined as anything that has some extent. For example, in
a 3d simulation, data on 2d surfaces is still considered volume data for the
purposes of observing data. The spectral coefficients can also be written as
volume data, though some care must be taken in that case to correctly identify
which mode is associated with which terms in the basis function
expansion. Whatever component will contribute volume data to be written must
register with the `observers::Observer` component (there currently isn't tested
support for directly registering with the `observers::ObserverWriter`). This
registration is the same as in the reduction data case.

Once the observers are registered, data is contributed to the
`observers::Observer` component using the
`observers::Actions::ContributeVolumeData` action. The data is packed into a
`std::vector<TensorComponent>`, where the `TensorComponent` is data from just
one tensor component or a reduction over a tensor. The `extents`,
`Spectral::Basis` and `Spectral::Quadrature` are currently also passed to the
`ContributeVolumeData` action. Once all the elements on a single core have
contributed their volume data to the `observers::Observer` group, the
`observers::Observer` group moves its data to the `observers::ObserverWriter`
component to be written. We write one file per node, appending the node ID to
the HDF5 file name to distinguish between files written by different nodes. The
HDF5 file name is specified in the input file using the
`observers::Tags::VolumeFileName` option. The data is written into a subfile of
the HDF5 file using the `h5::VolumeFile` class.

### Threading and NodeLocks

Since the `observers::ObserverWriter` class is a nodegroup, its entry methods
can be invoked simultaneously on different cores of the node. However, this can
lead to race conditions if care isn't taken. The biggest caution is that the
`DataBox` cannot be mutated on one core and simultaneously accessed on
another. This is because in order to guarantee a reasonable state for data in
the `DataBox`, it must be impossible to perform a `db::get` on a `DataBox` from
inside or while a `db::mutate` is being done. What this means in practice is
that all entry methods on a nodegroup must put their `DataBox` accesses inside
of a `node_lock.lock()` and `node_lock.unlock()` block. To achieve better
parallel performance and threading, the amount of work done while the entire
node is locked should be minimized. To this end, we have additional locks. One
for the HDF5 files because we do not require a threadsafe HDF5
(`observers::Tags::H5FileLock`). We also have locks for the objects mutated when
contributing reduction data (`observers::Tags::ReductionDataLock`) and the
objects mutated when contributing volume data
(`observers::Tags::VolumeDataLock`).

### Future changes
- It would be preferable to make the `Observer` and `ObserverWriter` parallel
  components more general and have them act as the core (node)group. Since any
  simple actions can be run on them, it should be possible to use them for most,
  if not all cases where we need a (node)group.
