{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Batched.Layers.Simple (biasNet, sigmoidNet, quadCostNet, averageGrad, reshapeNet) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.Network
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
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits

-- Bias

{-# NOINLINE biasForwardProgram #-}
biasForwardProgram :: KernelProgram '[TensorCL '[a], TensorCL '[ω, a], MTensorCL '[ω, a]]
biasForwardProgram = compile $ \ws xs ys -> do
  ~[u, i] <- traverse get_global_id [0,1]
  ys![u,i] .= ws![i] + xs![u,i]

biasForward :: (KnownNat ω, KnownNat (Prod ds)) => Tensor ds -> Tensor (ω ': ds) -> Tensor (ω ': ds)
biasForward ws xs = unsafePerformIO $ do
  ys <- Tensor.new
  clfun biasForwardProgram (dimVal ys) (Tensor.reshape ws) (Tensor.reshape xs) ys :: IO ()
  return (Tensor.reshape $ Tensor.unsafeFreeze ys)

biasNet :: forall ds ω m. (MonadIO m, KnownNat ω, KnownDims ds)
         => Network m ω (Prod ds) (Tensor (ω ': ds)) (Tensor (ω ': ds))
biasNet = network (Diff run) zero
  where
    run (BTensor ws :: BTensor ω ds, xs) = do
      return (biasForward ws xs, backward)
    backward dys = do
      return (dys, dys)

-- Sigmoid

sigmoidNet :: forall ds ω m. (MonadIO m, KnownDims ds)
           => Network m ω 0 (Tensor ds) (Tensor ds)
sigmoidNet = liftDiff Layers.sigmoidDiff

-- Quadratic Cost

{-# NOINLINE quadForwardProgram #-}
quadForwardProgram :: KernelProgram '[TensorCL '[ω, a], TensorCL '[ω, a], MTensorCL '[ω]]
quadForwardProgram = compile $ \xs ys os -> do
  u <- get_global_id 0
  let [_, a] = dimsOf xs
  acc <- eval 0
  forEach 0 a $ \i -> do
    d <- eval $ xs![u,i] - ys![u,i]
    acc .= acc + d^2
  os![u] .= acc

quadForward :: (KnownNat ω, KnownNat (Prod ds)) => Tensor (ω ': ds) -> Tensor (ω ': ds) -> Tensor '[ω]
quadForward xs ys = unsafePerformIO $ do
  os_flat <- Tensor.new
  clfun quadForwardProgram (dimVal os_flat) (Tensor.reshape xs) (Tensor.reshape ys) os_flat :: IO ()
  return (Tensor.reshape $ Tensor.unsafeFreeze os_flat)

{-# NOINLINE quadBackwardProgram #-}
quadBackwardProgram :: KernelProgram '[TensorCL '[ω], TensorCL '[ω, a], TensorCL '[ω, a], MTensorCL '[ω, a], MTensorCL '[ω, a]]
quadBackwardProgram = compile $ \dos xs ys dxs dys -> do
  ~[u,i] <- traverse get_global_id [0,1]
  d <- eval $ dos![u] * 2 * (xs![u,i] - ys![u,i])
  dxs![u,i] .= d
  dys![u,i] .= (-d)

quadBackward :: (KnownNat ω, KnownNat (Prod ds)) => Tensor '[ω] -> Tensor (ω ': ds) -> Tensor (ω ': ds) -> (Tensor (ω ': ds), Tensor (ω ': ds))
quadBackward dos xs ys = unsafePerformIO $ do
  dxs <- Tensor.new
  dys <- Tensor.new
  clfun quadBackwardProgram (dimVal dxs) dos (Tensor.reshape xs) (Tensor.reshape ys) dxs dys :: IO ()
  return (Tensor.reshape $ Tensor.unsafeFreeze dxs, Tensor.reshape $ Tensor.unsafeFreeze dys)

quadCostNet :: forall ds ω m. (MonadIO m, KnownNat ω, KnownDims ds)
            => Network m ω 0 (Tensor (ω ': ds), Tensor (ω ': ds)) (Tensor '[ω])
quadCostNet = liftDiff (Diff run)
  where
    run (xs, ys) = do
      return (quadForward xs ys, backward xs ys)
    backward xs ys dos = do
      return (quadBackward dos xs ys)

-- Reshape

reshapeNet :: forall as bs ω m. (KnownNat ω, Prod as ~ Prod bs, Monad m)
           => Network m ω 0 (Tensor (ω ': as)) (Tensor (ω ': bs))
reshapeNet = liftDiff (Diff run)
  where
    run as = return (Tensor.reshape as, backward)
    backward bs = return (Tensor.reshape bs)

-- Fanout

{-# NOINLINE averageProgram #-}
averageProgram :: KernelProgram '[TensorCL '[ω, a], MTensorCL '[a]]
averageProgram = compile $ \ws xs -> do
  i <- get_global_id 0
  let [ω, a] = dimsOf ws
  acc <- eval 0
  forEach 0 ω $ \u -> do
    acc .= acc + ws![u,i]
  xs![i] .= acc / castExpr ω

averageGrad :: (KnownNat ω, KnownNat (Prod ds)) => Tensor (ω ': ds) -> Tensor ds
averageGrad ws = unsafePerformIO $ do
  xs <- Tensor.new
  clfun averageProgram (dimVal xs) (Tensor.reshape ws) xs :: IO ()
  return (Tensor.reshape $ Tensor.unsafeFreeze xs)
