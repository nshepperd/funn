Funn: Functional Neural Networks in Haskell
===========================================

This is an experimental library exploring a combinator approach for building and training neural networks in haskell.

Traditional (eg. in C) libraries construct neural networks monolithically, by providing a comprehensive list of the layers' topologies to a function. The approach used in this library is an attempt at a composable system, in which networks are built by connecting smaller units together:

As `let one = fcLayer :: Network m (Blob 10) (Blob 20)` is a fully connected 10x20 layer, and `let two = sigmoidLayer :: Network m (Blob 20) (Blob 20)` is a sigmoid activation layer, we can compose them directly by feeding the output of the first into the second.

    one >>> two :: Network m (Blob 10) (Blob 20)

Parts of the interface are still quite ad-hoc and subject to change.

MHUG Talk
---------

The slides in `/mhug-talk-15` describe a mini talk I presented at the Melbourne haskell user group.
