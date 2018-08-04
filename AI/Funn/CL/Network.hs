{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Network (
  Network(..),
  params,
  first, second,
  liftDiff, network
  ) where


import           Control.Applicative
import           Control.Category
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.IORef
import           Data.List hiding (replicate)
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           GHC.TypeLits
import           Prelude hiding ((.), id)
import           System.IO.Unsafe

import           AI.Funn.CL.Buffer (Buffer)
import qualified AI.Funn.CL.Buffer as Buffer
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Param (Param)
import qualified AI.Funn.CL.Param as Param
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as T
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space

data Network m a b where
  Network :: KnownNat p => Proxy p -> Diff m (Param p, a) b -> RVar (Blob.Blob p) -> Network m a b

params :: Network m a b -> Int
params (Network p _ _) = fromIntegral (natVal p)

first :: (Monad m) => Network m a b -> Network m (a,c) (b,c)
first (Network p diff init) = Network p run init
  where
    run = Diff.assocL >>> Diff.first diff

second :: (Monad m) => Network m a b -> Network m (c,a) (c,b)
second (Network p diff init) = Network p run init
  where
    run = Diff.second Diff.swap >>> Diff.assocL >>> Diff.first diff >>> Diff.swap

liftDiff :: (MonadIO m) => Diff m a b -> Network m a b
liftDiff diff = Network (Proxy @ 0) run (pure (Blob.fromList []))
  where
    run = Diff (\(e, a) -> pure (a, \da -> do z <- zero
                                              pure (z, da))) >>> diff

network :: KnownNat p => Diff m (Param p, a) b -> RVar (Blob.Blob p) -> Network m a b
network diff init = Network Proxy diff init

network' :: (Monad m, Prod ds ~ p, KnownNat p) => Diff m (Tensor ds, a) b -> RVar (Blob.Blob p) -> Network m a b
network' diff init = Network Proxy (Diff run) init
  where
    run (par, a) = do
      (b, k) <- runDiff diff (Param.reshape par, a)
      return (b, backward k)
    backward k db = do
      (dpar, da) <- k db
      return (TL.fromStrict (T.reshape dpar), da)

concatInit :: (KnownNat a, KnownNat b)
           => RVar (Blob.Blob a) -> RVar (Blob.Blob b)
           -> RVar (Blob.Blob (a + b))
concatInit = liftA2 Blob.cat

connect :: MonadIO m => Network m a b -> Network m b c -> Network m a c
connect (Network p1 one i1) (Network p2 two i2) = Network Proxy (Diff run) init
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

    init = concatInit i1 i2


instance MonadIO m => Category (Network m) where
  id = liftDiff id
  (.) = flip connect
