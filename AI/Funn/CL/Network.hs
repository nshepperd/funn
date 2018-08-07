{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Network (
  Network(..),
  Indexed(..), Assoc(..),
  params, liftDiff, network,
  toDiff
  ) where


import           Control.Applicative
import           Control.Category
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import           GHC.TypeLits
import           Prelude hiding (id)
import           System.IO.Unsafe

import           AI.Funn.CL.Param (Param(..))
import qualified AI.Funn.CL.Param as Param
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space

data Network m p a b = Network {
  netDiff :: Diff m (Param p, a) b,
  netInit :: RVar (Blob.Blob p)
  }

params :: Network m p a b -> Proxy p
params _ = Proxy

network :: (Monad m, Prod ds ~ p, KnownNat p)
        => Diff m (Tensor ds, a) b
        -> RVar (Blob.Blob p)
        -> Network m p a b
network diff init = Network (Diff run) init
  where
    run (par, a) = do
      (b, k) <- runDiff diff (Param.reshape par, a)
      return (b, backward k)
    backward k db = do
      (dpar, da) <- k db
      return (TL.fromStrict (T.reshape dpar), da)

toParam :: (Monad m, KnownNat p)
        => Diff m (Tensor '[p]) (Param p)
toParam = Diff run
  where
    run input = return (Param input, backward)
    backward d_output = return (TL.toStrict d_output)

toDiff :: (Monad m, KnownNat p)
       => Network m p a b
       -> Diff m (Tensor '[p], a) b
toDiff (Network diff _) = Diff.first toParam >>> diff

connect :: (KnownNat i, KnownNat j, Monad m) => Network m i a b -> Network m j b c -> Network m (i+j) a c
connect (Network one i1) (Network two i2) = Network (Diff run) init
  where
    run (par, a) = do
      let (par1, par2) = Param.split par
      (b, k1) <- runDiff one (par1, a)
      (c, k2) <- runDiff two (par2, b)
      return (c, backward k1 k2)

    backward k1 k2 dc = do
      (dpar2, db) <- k2 dc
      (dpar1, da) <- k1 db
      return (TL.append dpar1 dpar2, da)

    init = liftA2 Blob.cat i1 i2

instance Monad m => Indexed (Network m) where
  iid = liftDiff id
  (~>>) = connect

liftDiff :: (Monad m) => Diff m a b -> Network m 0 a b
liftDiff diff = Network (Diff run) (pure (Blob.fromList []))
  where
    run (_, a) = do
      (b, k) <- runDiff diff a
      return (b, backward k)
    backward k db  = do
      da <- k db
      return (TL.nul, da)

pfirst :: (Monad m) => Network m p a b -> Network m p (a,c) (b,c)
pfirst (Network diff init) = Network (Diff run) init
  where
    run (p, (a,c)) = do
      (b, k) <- runDiff diff (p, a)
      return ((b,c), backward k)
    backward k (db, dc) = do
      (dp, da) <- k db
      return (dp, (da, dc))

psecond :: (Monad m) => Network m p a b -> Network m p (c,a) (c,b)
psecond (Network diff init) = Network (Diff run) init
  where
    run (p, (c,a)) = do
      (b, k) <- runDiff diff (p, a)
      return ((c,b), backward k)
    backward k (dc, db) = do
      (dp, da) <- k db
      return (dp, (dc, da))

pswap :: (Monad m) => Network m 0 (a,b) (b,a)
pswap = liftDiff Diff.swap

passocL :: (Monad m) => Network m 0 (a,(b,c)) ((a,b),c)
passocL = liftDiff Diff.assocL

passocR :: (Monad m) => Network m 0 ((a,b),c) (a,(b,c))
passocR = liftDiff Diff.assocR

instance Monad m => Assoc (Network m) where
  first = pfirst
  second = psecond
  swap = pswap
  assocL = passocL
  assocR = passocR
