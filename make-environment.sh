#!/bin/bash
exec cabal v2-build  --enable-profiling --write-ghc-environment-files=always
