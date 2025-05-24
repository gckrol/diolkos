# Diolkos Phase 2

Diolkos is currently using a star topology, with a root node and worker nodes. This is easy to implement and to reason about, but it has a lot of overhead. The main ones are:

- The time between when the calculations on the worker have been completed, to the time when this information is available again for other workers to use as inputs.
- Small hickups in receiving results from all nodes (e.g. scheduling, a lost packet) will prevent all nodes from starting any calculations.

The combination of these two things results in a fairly low utilization of the workers (~60%). These problems can be fixed by switching to a full mesh topology, and pipelining all calculations.

In a star topology the data flows like this:

    node -> switch -> root node -> switch -> node

Compare this to a (logical) full mesh:

    node -> switch -> node

Using a full mesh will reduce the effective latency between two nodes to half. 

Apart from just the topology changing, the computations can also be pipelined. The matrix multiplications are currently
being split by the output rows across all nodes. This will remain, but now the nodes will process their part of the matrix in tiles.
This means that when the tile is done the results can be sent to all other nodes back. This will increase the total amount of data that is sent, overall link utilization is not expected to be a problem. The effect of this pipelining is to hide a lot of latency.

This also means that all the nodes will run the entire inference logic. All the nodes know what's being calculated all the time, and when they need to do a matrix multiplication, they can do their part and 'magically' they will receive the rest of the results.

We'll use a custom protocol for all communication. This fits in with the 'low level' and 'minimal dependencies' ethos of the project. A standard like MPI isn't really suited anyway.

## FAQ

#### Would it be useful to use multicast/broadcast?

Not from a bandwidth perspective. The uplink to the switch from every node has plenty of capacity
available, as the downlink will have to contain data from all other nodes. This is because ethernet is
symmetric.

It could perhaps reduce the load on the node itself for sending all the data out, as for a broadcast/multicast packet
this would reduce the number of packets by a factor of (n-1). However, with the right optimizations this should not be a problem.