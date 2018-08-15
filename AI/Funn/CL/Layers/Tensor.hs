{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Layers.Tensor (reluNet, sigmoidDiff, sigmoidNet, biasNet, preluNet, quadCostNet, reshapeNet) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor(..), MTensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits


mappingProgram :: (Expr Double -> CL (Expr Double)) -> KernelProgram '[TensorCL '[d], MTensorCL '[d]]
mappingProgram f = compile $ \input output -> do
  i <- get_global_id 0
  v <- f (input![i])
  output![i] .= v

zippingProgram :: (Expr Double -> Expr Double -> CL (Expr Double)) -> KernelProgram '[TensorCL '[d], TensorCL '[d], MTensorCL '[d]]
zippingProgram f = compile $ \input1 input2 output -> do
  i <- get_global_id 0
  v <- f (input1![i]) (input2![i])
  output![i] .= v


mapProgram :: forall ds m. (MonadIO m, KnownDims ds)
           => KernelProgram '[TensorCL '[Prod ds], MTensorCL '[Prod ds]]
           -> Tensor ds -> m (Tensor ds)
mapProgram prog input = do
  output <- Tensor.new
  liftIO (clfun prog [dimSize output]
          (Tensor.reshape input)
          (Tensor.reshapeM output) :: IO ())
  return (Tensor.unsafeFreeze output)

zipProgram :: forall ds m. (MonadIO m, KnownDims ds)
           => KernelProgram '[TensorCL '[Prod ds], TensorCL '[Prod ds], MTensorCL '[Prod ds]]
           -> Tensor ds -> Tensor ds -> m (Tensor ds)
zipProgram prog input1 input2 = do
  output <- Tensor.new
  liftIO (clfun prog [dimSize output]
          (Tensor.reshape input1)
          (Tensor.reshape input2)
          (Tensor.reshapeM output) :: IO ())
  return (Tensor.unsafeFreeze output)

pointwiseDiff :: forall ds m. (MonadIO m, KnownDims ds)
              => KernelProgram '[TensorCL '[Prod ds], MTensorCL '[Prod ds]]
              -> KernelProgram '[TensorCL '[Prod ds], TensorCL '[Prod ds], MTensorCL '[Prod ds]]
              -> Diff m (Tensor ds) (Tensor ds)
pointwiseDiff fwd bwd = Diff run
  where
    run input = do
      output <- mapProgram fwd input
      return (output, zipProgram bwd input)

-- Relu

{-# NOINLINE reluForward #-}
reluForward :: KernelProgram '[TensorCL '[d], MTensorCL '[d]]
reluForward = mappingProgram $ \x -> return (fmax 0 x)

{-# NOINLINE reluBackward #-}
reluBackward :: KernelProgram '[TensorCL '[d], TensorCL '[d], MTensorCL '[d]]
reluBackward = zippingProgram $ \x dy -> return (fstep 0 x * dy)

reluDiff :: forall ds m. (MonadIO m, KnownDims ds)
         => Diff m (Tensor ds) (Tensor ds)
reluDiff = pointwiseDiff reluForward reluBackward

reluNet :: forall ds m. (MonadIO m, KnownDims ds)
         => Network m 0 (Tensor ds) (Tensor ds)
reluNet = liftDiff reluDiff

-- Sigmoid

{-# NOINLINE sigmoidForward #-}
sigmoidForward :: KernelProgram '[TensorCL '[d], MTensorCL '[d]]
sigmoidForward = mappingProgram $ \x -> do
  z <- eval (exp x)
  return $ z / (1 + z)

{-# NOINLINE sigmoidBackward #-}
sigmoidBackward :: KernelProgram '[TensorCL '[d], TensorCL '[d], MTensorCL '[d]]
sigmoidBackward = zippingProgram $ \x dy -> do
  z <- eval $ exp (-abs x)
  return $ dy * z / (1 + z)^2

sigmoidDiff :: forall ds m. (MonadIO m, KnownDims ds)
            => Diff m (Tensor ds) (Tensor ds)
sigmoidDiff = pointwiseDiff sigmoidForward sigmoidBackward

sigmoidNet :: forall ds m. (MonadIO m, KnownDims ds)
         => Network m 0 (Tensor ds) (Tensor ds)
sigmoidNet = liftDiff sigmoidDiff

-- Bias

biasDiff :: forall ds m. (MonadIO m, KnownDims ds)
         => Diff m (Tensor ds, Tensor ds) (Tensor ds)
biasDiff = Diff run
  where
    run (p, input) = do
      output <- plus p input
      return (output, backward)
    backward d_output = return (d_output, d_output)

biasNet :: forall ds m. (MonadIO m, KnownDims ds)
         => Network m (Prod ds) (Tensor ds) (Tensor ds)
biasNet = network biasDiff zero

-- Parametric Relu

{-# NOINLINE preluForward #-}
preluForward :: KernelProgram '[TensorCL '[n], TensorCL '[n], MTensorCL '[n]]
preluForward = compile $ \rs xs ys -> do
  i <- get_global_id 0
  let
    r = rs![i]
    x = xs![i]
  ys![i] .= x * (r + (1 - r) * fstep 0 x)

{-# NOINLINE preluBackward #-}
preluBackward :: KernelProgram '[TensorCL '[n], TensorCL '[n], TensorCL '[n], MTensorCL '[n]]
preluBackward = compile $ \rs xs dys dxs -> do
  i <- get_global_id 0
  let
    r = rs![i]
    x = xs![i]
    dy = dys![i]
  dxs![i] .= dy * (r + (1 - r) * fstep 0 x)

{-# NOINLINE preluBackpar #-}
preluBackpar :: KernelProgram '[TensorCL '[n], TensorCL '[n], MTensorCL '[n]]
preluBackpar = compile $ \xs dys drs -> do
  i <- get_global_id 0
  let
    x = xs![i]
    dy = dys![i]
  drs![i] .= dy * fmin 0 x

preluNet :: forall ds m. (MonadIO m, KnownDims ds)
         => Network m (Prod ds) (Tensor ds) (Tensor ds)
preluNet = network (Diff run) (Blob.generate (pure 1))
  where
    run (rs, xs) = do
      ys <- Tensor.new
      liftIO (clfun preluForward [dimSize ys]
              rs
              (Tensor.reshape xs)
              (Tensor.reshapeM ys) :: IO ())
      return (Tensor.unsafeFreeze ys, backward rs xs)
    backward rs xs dys = do
      dxs <- Tensor.new
      liftIO (clfun preluBackward [dimSize dxs]
              rs
              (Tensor.reshape xs)
              (Tensor.reshape dys)
              (Tensor.reshapeM dxs) :: IO ())
      drs <- Tensor.new
      liftIO (clfun preluBackpar [dimSize drs]
              (Tensor.reshape xs)
              (Tensor.reshape dys)
              drs :: IO ())
      return (Tensor.unsafeFreeze drs, Tensor.unsafeFreeze dxs)


-- Quadratic Cost

quadCostNet :: forall ds m. (MonadIO m, KnownDims ds)
            => Network m 0 (Tensor ds, Tensor ds) Double
quadCostNet = liftDiff (Diff run)
  where
    run (xs, ys) = do
      diff <- Tensor.subTensor xs ys
      diff2 <- Tensor.squareTensor diff
      o <- sum <$> Tensor.toList diff2
      return (o, backward diff)
    backward diff d = do
      dxs <- scale (2*d) diff
      dys <- scale (-2*d) diff
      return (dxs, dys)

-- Reshape

reshapeNet :: forall ds es m. (MonadIO m, Prod ds ~ Prod es)
           => Network m 0 (Tensor ds) (Tensor es)
reshapeNet = liftDiff (Diff run)
  where
    run xs = return (Tensor.reshape xs, backward)
    backward dys = return (Tensor.reshape dys)
