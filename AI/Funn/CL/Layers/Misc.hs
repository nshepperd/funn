{-# LANGUAGE ConstraintKinds #-}
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
module AI.Funn.CL.Layers.Misc where

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
import           AI.Funn.CL.Tensor (Tensor, MTensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.Space
import           AI.Funn.TypeLits

{-# NOINLINE fillSquare #-}
fillSquare :: KernelProgram '[MTensorCL [w,h]]
fillSquare = compile $ \arr -> do
  [x, y] <- traverse get_global_id [0,1]
  arr ! [x,y] .= castExpr x + castExpr y

{-# NOINLINE iconv2dForward #-}
iconv2dForward :: KernelProgram '[TensorCL [k, k, c1, c2],
                                  TensorCL [ow + k - 1, oh + k - 1, c1],
                                  MTensorCL [ow, oh, c2]]
iconv2dForward = compile $ \flt input output -> do
  [ox, oy, oc] <- traverse get_global_id [0, 1, 2]
  let [k, _, c1, c2] = dimsOf flt
  acc <- eval 0
  forEach 0 k $ \kx -> do
    forEach 0 k $ \ky -> do
      forEach 0 c1 $ \ic -> do
        acc .= acc + flt![kx, ky, ic, oc] * input![ox + kx, oy + ky, ic]
  output![ox, oy, oc] .= acc

{-# NOINLINE iconv2dBackward #-}
iconv2dBackward :: KernelProgram '[TensorCL [k, k, c1, c2],
                                  MTensorCL [ow + k - 1, oh + k - 1, c1],
                                  TensorCL [ow, oh, c2]]
iconv2dBackward = compile $ \flt d_input d_output -> do
  [ix, iy, ic] <- traverse get_global_id [0, 1, 2]
  let [kw, kh, c1, c2] = dimsOf flt
  let [iw, ih, _] = dimsOf d_input
  acc <- eval 0
  kernel_x1 <- eval $ fmax 0 (kw - (iw - ix))
  kernel_x2 <- eval $ fmin kw (ix + 1)
  kernel_y1 <- eval $ fmax 0 (kh - (ih - iy))
  kernel_y2 <- eval $ fmin kh (iy + 1)
  forEach kernel_x1 kernel_x2 $ \kx -> do
    forEach kernel_y1 kernel_y2 $ \ky -> do
      forEach 0 c2 $ \oc -> do
        acc .= acc + flt![kx, ky, ic, oc] * d_output![ix - kx, iy - ky, oc]
  d_input![ix, iy, ic] .= acc

{-# NOINLINE iconv2dFilter #-}
iconv2dFilter :: KernelProgram '[MTensorCL [k, k, c1, c2],
                                 TensorCL [ow + k - 1, oh + k - 1, c1],
                                 TensorCL [ow, oh, c2]]
iconv2dFilter = compile $ \d_flt input d_output -> do
  [kx, ky, ic, oc] <- get_global_id 0 >>= splitIndex (dimsOf d_flt)
  let [ow, oh, _] = dimsOf d_output
  acc <- eval 0
  forEach 0 ow $ \ox -> do
    forEach 0 oh $ \oy -> do
      acc .= acc + input![ox + kx, oy + ky, ic] * d_output![ox, oy, oc]
  d_flt![kx, ky, ic, oc] .= acc

iconv2dDiff :: forall k ow oh c1 c2 m. (MonadIO m, KnownDims [k, ow, oh, c1, c2],
                                        (1 <=? k) ~ 'True)
            => Diff m
               (Tensor [k, k, c1, c2], Tensor [ow + k - 1, oh + k - 1, c1])
               (Tensor [ow, oh, c2])
iconv2dDiff = Diff run
  where
    run (flt, input) = do
      output <- Tensor.new @ [ow, oh, c2]
      liftIO (clfun iconv2dForward (dimVal output) flt input output :: IO ())
      return (Tensor.unsafeFreeze output, backward flt input)

    backward flt input d_output = do
      d_input <- Tensor.new
      liftIO (clfun iconv2dBackward (dimVal d_input) flt d_input d_output :: IO ())
      d_flt <- Tensor.new
      liftIO (clfun iconv2dFilter [dimSize d_flt] d_flt input d_output :: IO ())
      return (Tensor.unsafeFreeze d_flt, Tensor.unsafeFreeze d_input)
