\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Parallelization in SpECTRE {#tutorial_parallel_concepts}

This overview describes the concepts and terminology that SpECTRE uses
to enable parallelism.  This overview is a general discussion with no
code examples.  Subsequent tutorials will provide a more in-depth
exploration of the parallelization infrastructure, including code
examples.

Unlike many parallel scientific codes which use data-based
parallelism, SpECTRE uses task-based parallelism.  The classical
strategy for parallelism (data-based parallelism) is to assign a
portion of the data to processes (or threads) that synchronously
execute compute kernels.  This is implemented in many codes but it is
difficult to design codes with this strategy that will efficiently
scale for complex multi-scale, multi-physics workloads. Task-based
parallelism provides a solution: Instead of dividing work between
parallel processes based on data ownership, there is a set of tasks
and their inter-dependencies. Tasks are scheduled and assigned to
processes dynamically, providing opportunities for load balancing and
minimization of idle threads.  By dividing the program into small
enough tasks such that you have several tasks per thread,
communication time is hidden by interleaving tasks that are ready to
be executed with tasks that are waiting for data.

In order to implement task-based parallelism, SpECTRE is built on top
of the parallel programming framework of the Charm++ library, which is
developed by the [Parallel Programming
Laboratory](http://charm.cs.illinois.edu/) at the University of
Illinois.  Charm++ is a mature parallel programming framework that
provides intra-node threading and can use a variety of communication
interfaces (including MPI) to communicate between nodes.  Charm++ has
a large user base, which includes users of the cosmological
\f$N\f$-body code
[ChaNGa](https://github.com/N-BodyShop/changa/wiki/ChaNGa) and of the
molecular dynamics code
[NAMD](https://www.ks.uiuc.edu/Research/namd/).

## Charm++ basic concepts

In order to understand how parallelization works in SpECTRE, it is
useful to understand the basic concepts in the design of Charm++.
Much of the following is quoted verbatim from the [Charm++
documentation](https://charm.readthedocs.io), interspersed with
comments on how SpECTRE interacts with Charm++.

> Charm++ is a C++-based parallel programming system, founded on the
> migratable-objects programming model, and supported by a novel and
> powerful adaptive runtime system. It supports both irregular as well
> as regular applications, and can be used to specify task-parallelism
> as well as data parallelism in a single application. It automates
> dynamic load balancing for task-parallel as well as data-parallel
> applications, via separate suites of load-balancing strategies. Via
> its message-driven execution model, it supports automatic latency
> tolerance, modularity and parallel composition. Charm++ also supports
> automatic checkpoint/restart, as well as fault tolerance based on
> distributed checkpoints.

SpECTRE currently wraps only some of the features of Charm++,
primarily the ones that support task-parallelism.  We are just
beginning our exploration of dynamic load balancing.  Coming soon we
will utilize automatic checkpoint/restart.  At present we do not use
Charm++ support for fault tolerance.

> The key feature of the migratable-objects programming model is
> over-decomposition: The programmer decomposes the program into a
> large number of work units and data units, and specifies the
> computation in terms of creation of and interactions between these
> units, without any direct reference to the processor on which any
> unit resides. This empowers the runtime system to assign units to
> processors, and to change the assignment at runtime as necessary.

SpECTRE's parallelization module is designed to make it easy for users
to exploit the migratable-object model by providing a framework to
define the units into which a program can be decomposed.

> A basic unit of parallel computation in Charm++ programs is a
> chare.  At its most basic level, it is just a C++ object. A Charm++
> computation consists of a large number of chares distributed on
> available processors of the system, and interacting with each other
> via asynchronous method invocations. Asynchronously invoking a
> method on a remote object can also be thought of as sending a
> “message” to it. So, these method invocations are sometimes referred
> to as messages. (besides, in the implementation, the method
> invocations are packaged as messages anyway). Chares can be created
> dynamically.

In SpECTRE, we wrap Charm++ chares in struct templates that we call
__parallel components__ that represent a collection of distributed
objects.  We wrap the asynchronous method invocations between the
elements of parallel components in struct templates that we call
__actions__.  Thus, each element of a parallel component can be
thought of as a C++ object that exists on one core on the
supercomputer, and an action as calling a member function of that
object, even if the caller is on another core.

> Conceptually, the system maintains a “work-pool” consisting of seeds
> for new chares, and messages for existing chares. The Charm++
> runtime system (Charm RTS) may pick multiple items,
> non-deterministically, from this pool and execute them, with the
> proviso that two different methods cannot be simultaneously
> executing on the same chare object (say, on different
> processors). Although one can define a reasonable theoretical
> operational semantics of Charm++ in this fashion, a more practical
> description of execution is useful to understand Charm++. A Charm++
> application’s execution is distributed among Processing Elements
> (PEs), which are OS threads or processes depending on the selected
> Charm++ build options. On each PE, there is a scheduler operating
> with its own private pool of messages. Each instantiated chare has
> one PE which is where it currently resides. The pool on each PE
> includes messages meant for chares residing on that PE, and seeds
> for new chares that are tentatively meant to be instantiated on that
> PE. The scheduler picks a message, creates a new chare if the
> message is a seed (i.e. a constructor invocation) for a new chare,
> and invokes the method specified by the message. When the method
> returns control back to the scheduler, it repeats the cycle
> (i.e. there is no pre-emptive scheduling of other invocations).

It is very important to keep in mind that the actions that are
executed on elements of parallel components are done so
non-deterministically by the run-time system.  Therefore it is the
responsibility of the programmer to ensure that actions are not called
out of order.  This means that if action B must be executed after
action A on a given element of a parallel component, the programmer
must ensure that either action B is called after the completion of
action A (i.e. it is not sufficient that action B is invoked after
action A is invoked), or that a `is_ready` function of action B only
succeeds if action A has been completed.

> When a chare method executes, it may create method invocations for
> other chares. The Charm Runtime System (RTS) locates the PE where
> the targeted chare resides, and delivers the invocation to the
> scheduler on that PE.

In SpECTRE, this is done by one element of a parallel component
calling an action on an element of a another parallel component.

> Methods of a chare that can be remotely invoked are called entry
> methods. Entry methods may take serializable parameters, or a
> pointer to a message object. Since chares can be created on remote
> processors, obviously some constructor of a chare needs to be an
> entry method. Ordinary entry methods are completely non-preemptive-
> Charm++ will not interrupt an executing method to start any other
> work, and all calls made are asynchronous.

In SpECTRE, the struct template that defines a parallel component has
taken care of creating the entry methods for the underlying chare,
which are then called by invoking actions.

> Charm++ provides dynamic seed-based load balancing. Thus location
> (processor number) need not be specified while creating a remote
> chare. The Charm RTS will then place the remote chare on a suitable
> processor. Thus one can imagine chare creation as generating only a
> seed for the new chare, which may take root on some specific
> processor at a later time.

We are just in the process of beginning to explore the load-balancing
features of Charm++, but plan to have new elements of parallel
components (the wrapped chares) be creatable using actions, without
specifying the location on which the new element is created.

> Chares can be grouped into collections. The types of collections of
> chares supported in Charm++ are: chare-arrays, chare-groups, and
> chare-nodegroups, referred to as arrays, groups, and nodegroups
> throughout this manual for brevity. A Chare-array is a collection of
> an arbitrary number of migratable chares, indexed by some index
> type, and mapped to processors according to a user-defined map
> group. A group (nodegroup) is a collection of chares, with exactly
> one member element on each PE (“node”).

Each of SpECTRE's parallel components has a type alias `chare_type`
corresponding to whether it is a chare-array, chare-group, or
chare-nodegroup.  In addition we support a singleton which is
essentially a one-element array.

> Charm++ does not allow global variables, except readonly
> variables. A chare can normally only access its own data
> directly. However, each chare is accessible by a globally valid
> name. So, one can think of Charm++ as supporting a global object
> space.

SpECTRE does not use the readonly global variables provided by
Charm++.  Instead SpECTRE provides a nodegroup called the
`ConstGlobalCache` which provides global access to read-only objects,
as well as a way to access every parallel component.

> Every Charm++ program must have at least one mainchare. Each
> mainchare is created by the system on processor 0 when the Charm++
> program starts up. Execution of a Charm++ program begins with the
> Charm RTS constructing all the designated mainchares. For a
> mainchare named X, execution starts at constructor X() or X(CkArgMsg
> *) which are equivalent. Typically, the mainchare constructor starts
> the computation by creating arrays, other chares, and groups. It can
> also be used to initialize shared readonly objects.

SpECTRE provides a pre-defined mainchare called `Main` that is run
when a SpECTRE executable is started.  `Main` will create the other
parallel components, and initialize items in the `ConstGlobalCache`
whose items can be used by any parallel component.

> Charm++ program execution is terminated by the CkExit call. Like the
> exit system call, CkExit never returns, and it optionally accepts an
> integer value to specify the exit code that is returned to the
> calling shell. If no exit code is specified, a value of zero
> (indicating successful execution) is returned. The Charm RTS ensures
> that no more messages are processed and no entry methods are called
> after a CkExit. CkExit need not be called on all processors; it is
> enough to call it from just one processor at the end of the
> computation.

SpECTRE wraps `CkExit` with the function `Parallel::exit`.  As no more
messages are processed by Charm++ after this call, SpECTRE also
defines a special `Exit` phase that is guaranteed to be executed after
all messages and entry methods have been processed.

> As described so far, the execution of individual Chares is
> “reactive”: When method A is invoked the chare executes this code,
> and so on. But very often, chares have specific life-cycles, and the
> sequence of entry methods they execute can be specified in a
> structured manner, while allowing for some localized non-determinism
> (e.g. a pair of methods may execute in any order, but when they both
> finish, the execution continues in a pre-determined manner, say
> executing a 3rd entry method).

Charm++ provides a special notation to simplify expression of such
control structures, but this requires writing specialized interface
files that are parsed by Charm++.  SpECTRE does not support this;
rather we split the executable into a set of user-defined phases. In
each phase, each parallel component will execute a user-defined list
of actions.

> The normal entry methods, being asynchronous, are not allowed to
> return any value, and are declared with a void return type.

SpECTRE's actions do not return any value to the calling component.
Instead when the action is finished it can call another action to send
data to an element of any parallel component.

> To support asynchronous method invocation and global object space,
> the RTS needs to be able to serialize (“marshall”) the parameters,
> and be able to generate global “names” for chares. For this purpose,
> programmers have to declare the chare classes and the signature of
> their entry methods in a special “.ci” file, called an interface
> file. Other than the interface file, the rest of a Charm++ program
> consists of just normal C++ code. The system generates several
> classes based on the declarations in the interface file, including
> “Proxy” classes for each chare class. Those familiar with various
> component models (such as CORBA) in the distributed computing world
> will recognize “proxy” to be a dummy, standin entity that refers to
> an actual entity. For each chare type, a “proxy” class exists. The
> methods of this “proxy” class correspond to the remote methods of
> the actual class, and act as “forwarders”. That is, when one invokes
> a method on a proxy to a remote object, the proxy marshalls the
> parameters into a message, puts adequate information about the
> target chare on the envelope of the message, and forwards it to the
> remote object. Individual chares, chare array, groups, node-groups,
> as well as the individual elements of these collections have a such
> a proxy. Multiple methods for obtaining such proxies are described
> in the manual. Proxies for each type of entity in Charm++ have some
> differences among the features they support, but the basic syntax
> and semantics remain the same - that of invoking methods on the
> remote object by invoking methods on proxies.

SpECTRE has wrapped all of this functionality in order to make it
easier to use.  SpECTRE automatically creates the interface files for
each parallel component using template metaprogramming.  SpECTRE
provides proxies for each parallel component that are all held in the
`ConstGlobalCache` which is available to every parallel component.  In
order for actions to be called as entry methods on remote parallel
components, the arguments to the function call must be serializable.
Charm++ provides the `PUP` framework to serialize objects where `PUP`
stands for pack-unpack.  Since the PUP framework is used by Charm++
for checkpointing, load-balancing, and passing arguments when calling
actions, all user-defined classes with member data must define a `pup`
function.

> In terms of physical resources, we assume the parallel machine
> consists of one or more nodes, where a node is a largest unit over
> which cache coherent shared memory is feasible (and therefore, the
> maximal set of cores per which a single process can run. Each node
> may include one or more processor chips, with shared or private
> caches between them. Each chip may contain multiple cores, and each
> core may support multiple hardware threads (SMT for example).
> Charm++ recognizes two logical entities: a PE (processing element)
> and a logical node, or simply “node”. In a Charm++ program, a PE is
> a unit of mapping and scheduling: each PE has a scheduler with an
> associated pool of messages. Each chare is assumed to reside on one
> PE at a time. A logical node is implemented as an OS process. In
> non-SMP mode there is no distinction between a PE and a logical
> node. Otherwise, a PE takes the form of an OS thread, and a logical
> node may contain one or more PEs. Physical nodes may be partitioned
> into one or more logical nodes. Since PEs within a logical node
> share the same memory address space, the Charm++ runtime system
> optimizes communication between them by using shared
> memory. Depending on the runtime command-line parameters, a PE may
> optionally be associated with a subset of cores or hardware threads.

In other words, how Charm++ defines a node and a PE depends upon how
Charm++ was installed on a system.  The executable
`Executables/ParallelInfo` can be used to determine how many nodes and
PEs exist for a given Charm++ build and the runtime command-line
parameters passed when calling the executable.

For more details see the [Charm++
documentation](https://charm.readthedocs.io) and \ref
dev_guide_parallelization_foundations.
