{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Layers.Tensor (reluNet, sigmoidNet, biasNet, quadCostNet) where

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
