{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PartialTypeSignatures #-}
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
module AI.Funn.CL.Layers.Convolution where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import           GHC.Stack
import           GHC.TypeLits

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.Layers.Tensor
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor, MTensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits


-- ox + kx - pad >= 0
-- kx >= pad - ox

-- ox + kx - pad < iw
-- ox + kx < iw + pad
-- kx < iw + pad - ox

{-# NOINLINE conv2dForward #-}
conv2dForward :: KernelProgram '[Expr Int,
                                 TensorCL [k, k, c1, c2],
                                 TensorCL [iw, ih, c1],
                                 MTensorCL [ow, oh, c2]]
conv2dForward = compile $ \pad flt input output -> do
  ~[ox, oy, oc] <- traverse get_global_id [0, 1, 2]
  let
    [k, _, c1, c2] = dimsOf flt
    [iw, ih, _] = dimsOf input
    [ow, oh, _] = dimsOf output
  acc <- eval 0
  kernel_x1 <- eval (fmax 0 (pad - ox))
  kernel_x2 <- eval (fmin k (iw + pad - ox))
  kernel_y1 <- eval (fmax 0 (pad - oy))
  kernel_y2 <- eval (fmin k (ih + pad - oy))
  forEach kernel_x1 kernel_x2 $ \kx -> do
    forEach kernel_y1 kernel_y2 $ \ky -> do
      forEach 0 c1 $ \ic -> do
        acc .= acc + flt![kx, ky, ic, oc] * input![ox + kx - pad, oy + ky - pad, ic]
  output![ox, oy, oc] .= acc


-- 0 <= ix + pad - kx
-- kx <= ix + pad
-- kx < ix + pad + 1

-- ow > ix + pad - kx
-- ow + kx > ix + pad
-- kx > ix + pad - ow
-- kx >= ix + pad - ow + 1

{-# NOINLINE conv2dBackward #-}
conv2dBackward :: KernelProgram '[Expr Int,
                                  TensorCL [k, k, c1, c2],
                                  MTensorCL [iw, ih, c1],
                                  TensorCL [ow, oh, c2]]
conv2dBackward = compile $ \pad flt d_input d_output -> do
  ~[ix, iy, ic] <- traverse get_global_id [0, 1, 2]
  let
    [k, _, c1, c2] = dimsOf flt
    [iw, ih, _] = dimsOf d_input
    [ow, oh, _] = dimsOf d_output
  acc <- eval 0
  kernel_x1 <- eval $ fmax 0 (ix + pad - ow + 1)
  kernel_x2 <- eval $ fmin k (ix + pad + 1)
  kernel_y1 <- eval $ fmax 0 (iy + pad - oh + 1)
  kernel_y2 <- eval $ fmin k (iy + pad + 1)
  forEach kernel_x1 kernel_x2 $ \kx -> do
    forEach kernel_y1 kernel_y2 $ \ky -> do
      forEach 0 c2 $ \oc -> do
        acc .= acc + flt![kx, ky, ic, oc] * d_output![ix + pad - kx, iy + pad - ky, oc]
  d_input![ix, iy, ic] .= acc


-- ox + kx - pad >= 0
-- ox >= pad - kx

-- ox + kx - pad < iw
-- ox < iw + pad - kx

{-# NOINLINE conv2dFilter #-}
conv2dFilter :: KernelProgram '[Expr Int,
                                MTensorCL [k, k, c1, c2],
                                TensorCL [iw, ih, c1],
                                TensorCL [ow, oh, c2]]
conv2dFilter = compile $ \pad d_flt input d_output -> do
  ~[kx, ky, ic, oc] <- get_global_id 0 >>= splitIndex (dimsOf d_flt)
  let
    [iw, ih, _] = dimsOf input
    [ow, oh, _] = dimsOf d_output
  acc <- eval 0
  ox1 <- eval (fmax 0 (pad - kx))
  ox2 <- eval (fmin ow (iw + pad - kx))
  oy1 <- eval (fmax 0 (pad - ky))
  oy2 <- eval (fmin oh (ih + pad - ky))
  forEach ox1 ox2 $ \ox -> do
    forEach oy1 oy2 $ \oy -> do
      acc .= acc + input![ox + kx - pad, oy + ky - pad, ic] * d_output![ox, oy, oc]
  d_flt![kx, ky, ic, oc] .= acc

conv2dDiff :: forall k pad iw ih c1 c2 m proxy.
              (MonadIO m, KnownDimsF [k, pad, iw, ih, c1, c2], (1 <= k), k <= iw, k <= ih)
            => proxy pad -> Diff m
              (Tensor [k, k, c1, c2], Tensor [iw, ih, c1])
              (Tensor [iw + 2 * pad - k + 1,
                       ih + 2 * pad - k + 1,
                       c2])
conv2dDiff _ = Diff run
  where
    run (flt, input) = do
      output <- Tensor.new
      liftIO (clfun conv2dForward (dimVal output) pad flt input output :: IO ())
      return (Tensor.unsafeFreeze output, backward flt input)

    backward flt input d_output = do
      d_input <- Tensor.new
      liftIO (clfun conv2dBackward (dimVal d_input) pad flt d_input d_output :: IO ())
      d_flt <- Tensor.new
      liftIO (clfun conv2dFilter [dimSize d_flt] pad d_flt input d_output :: IO ())
      return (Tensor.unsafeFreeze d_flt, Tensor.unsafeFreeze d_input)

    pad = fromIntegral (natVal (Proxy @ pad)) :: Int

-- Adds bias.
conv2d :: forall k pad iw ih c1 c2 m proxy.
          (MonadIO m, KnownDimsF [k, pad, iw, ih, c1, c2], (1 <= k), k <= iw, k <= ih)
       => proxy pad
       -> Network m _ (Tensor [iw, ih, c1]) (Tensor [iw + 2 * pad - k + 1,
                                                     ih + 2 * pad - k + 1,
                                                     c2])
conv2d _ = network (conv2dDiff (Proxy @ pad)) init ~>> biasNet
  where
    init = Blob.generate (normal 0 (1 / sqrt d))
    d = k * k * sqrt (c1 * c2)
    [k, c1, c2] = map fromIntegral (dimVal (Proxy @[k, c1, c2]))

conv3x3 :: forall w h c1 c2 m.
           (MonadIO m, KnownDimsF [w, h, c1, c2], 3 <= w, 3 <= h)
        => Network m _ (Tensor [w, h, c1]) (Tensor [w, h, c2])
conv3x3 = conv2d @3 @1 Proxy
