{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Batched.Layers.GLOW (reshapeInv) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe
import           Text.Printf

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.GLOW (Invertible(..), invert)
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Network (Network(..), liftDiff)
import           AI.Funn.CL.Batched.Param (Param(..))
import qualified AI.Funn.CL.Batched.Param as Param
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import qualified AI.Funn.CL.Layers.Tensor as Layers
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor(..), MTensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits

reshapeInv :: (KnownNat ω, Prod as ~ Prod bs, Monad m)
           => Invertible m ω 0 (Tensor (ω ': as)) (Tensor (ω ': bs))
reshapeInv = Invertible reshapeNet reshapeNet

reshapeInvLazy :: (KnownNat ω, Prod as ~ Prod bs, Monad m)
               => Invertible m ω 0 (TL.Tensor (ω ': as)) (TL.Tensor (ω ': bs))
reshapeInvLazy = Invertible (liftDiff (Diff run)) (liftDiff (Diff run))
  where
    run as = return (TL.reshape as, backward)
    backward bs = return (TL.reshape bs)

splitInvLazy :: (KnownDimsF [ω, a, b], Monad m)
             => Invertible m ω 0 (TL.Tensor [ω, a+b]) (TL.Tensor [ω, a], TL.Tensor [ω, b])
splitInvLazy = Invertible split append
  where
    split = liftDiff (Diff $ \ab -> return (TL.splitW ab, return . uncurry TL.appendW))
    append = liftDiff (Diff $ \(a,b) -> return (TL.appendW a b, return . TL.splitW))

appendInvLazy :: (KnownDimsF [ω, a, b], Monad m)
             => Invertible m ω 0 (TL.Tensor [ω, a], TL.Tensor [ω, b]) (TL.Tensor [ω, a+b])
appendInvLazy = invert splitInvLazy
