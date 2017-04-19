{-# LANGUAGE TypeFamilies, FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE PartialTypeSignatures #-}
module AI.Funn.Network.RNN (scanlLayer, mapLayer, zipLayer, unzipLayer, vsumLayer) where

import           Control.Applicative
import           Control.Applicative.Backwards
import           Control.Monad
import           Control.Monad.State.Lazy
import           Data.Foldable
import           Data.Traversable

import           Data.Coerce
import           Debug.Trace

import           Foreign.C
import           Foreign.Ptr
import           System.IO
import           System.IO.Unsafe

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V

import           Data.Functor.Identity

import           AI.Funn.Diff.Diff (Derivable(..), Diff(..), Additive(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Network.Network
import           AI.Funn.Diff.RNN

scanlLayer :: forall m s i o. (Monad m)
           => Network m (s,i) (s, o) -> Network m (s, Vector i) (s, Vector o)
scanlLayer (Network p sub_diff sub_init) = Network p diff sub_init
  where
    diff = scanlDiff sub_diff

zipLayer :: (Monad m) => Network m (Vector x, Vector y) (Vector (x,y))
zipLayer = liftDiff zipDiff

unzipLayer :: (Monad m) => Network m (Vector (x,y)) (Vector x, Vector y)
unzipLayer = liftDiff unzipDiff

mapLayer :: forall m i o. (Monad m) => Network m i o -> Network m (Vector i) (Vector o)
mapLayer (Network p sub_diff sub_init) = Network p (Diff run) sub_init
  where
    run (pars, is) = do
      oks <- traverse (\i -> runDiff sub_diff (pars,i)) is
      let
        (os, ks) = V.unzip oks
      return (os, backward ks)

    backward ks dos = do
      dpis <- sequenceA (V.zipWith id ks dos)
      let
        (dps, dis) = V.unzip dpis
      return (runIdentity (plusm dps), dis)

vsumLayer :: (Monad m, Additive m a) => Network m (Vector a) a
vsumLayer = liftDiff vsumDiff
