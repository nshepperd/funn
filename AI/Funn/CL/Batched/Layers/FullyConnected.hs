{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Batched.Layers.FullyConnected (fcNet) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Network
import           AI.Funn.CL.Batched.Param (Param(..))
import qualified AI.Funn.CL.Batched.Param as Param
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor(..), MTensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Space
import           AI.Funn.TypeLits

{-# NOINLINE mulMVProgram #-}
mulMVProgram :: KernelProgram '[TensorCL '[b, a], TensorCL '[ω, a], MTensorCL '[ω, b]]
mulMVProgram = compile $ \ws xs ys -> do
  ~[u, j] <- traverse get_global_id [0,1]
  let [_, a] = dimsOf xs
  acc <- eval 0
  forEach 0 a $ \i -> do
    acc .= acc + ws![j,i] * xs![u, i]
  ys![u, j] .= acc

mulMV :: KnownDimsF [ω, a, b] => Tensor [b,a] -> Tensor '[ω, a] -> Tensor '[ω, b]
mulMV ws xs = unsafePerformIO $ do
  ys <- Tensor.new
  clfun mulMVProgram (dimVal ys) ws xs ys :: IO ()
  return (Tensor.unsafeFreeze ys)

{-# NOINLINE mulVMProgram #-}
mulVMProgram :: KernelProgram '[TensorCL '[ω, b], TensorCL '[b, a], MTensorCL '[ω, a]]
mulVMProgram = compile $ \ys ws xs -> do
  ~[u, i] <- traverse get_global_id [0,1]
  let [_, b] = dimsOf ys
  acc <- eval 0
  forEach 0 b $ \j -> do
    acc .= acc + ws![j,i] * ys![u, j]
  xs![u, i] .= acc

mulVM :: KnownDimsF [ω, a, b] => Tensor '[ω, b] -> Tensor [b,a] -> Tensor '[ω, a]
mulVM ys ws = unsafePerformIO $ do
  xs <- Tensor.new
  clfun mulVMProgram (dimVal xs) ys ws xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE outerVVProgram #-}
outerVVProgram :: KernelProgram '[TensorCL '[ω, a], TensorCL '[ω, b], MTensorCL '[ω, a, b]]
outerVVProgram = compile $ \xs ys ws -> do
  ~[u,i,j] <- traverse get_global_id [0,1,2]
  ws![u,i,j] .= xs![u,i] * ys![u,j]

outerVV :: KnownDimsF [ω, a, b] => Tensor '[ω, a] -> Tensor '[ω, b] -> Tensor [ω,a,b]
outerVV xs ys = unsafePerformIO $ do
  ws <- Tensor.new
  clfun outerVVProgram (dimVal ws) xs ys ws :: IO ()
  return (Tensor.unsafeFreeze ws)

fcNet :: forall a b ω m. (MonadIO m, KnownDimsF [ω,a,b])
      => Network m ω _ (Tensor '[ω, a]) (Tensor '[ω, b])
fcNet = network (Diff run) (Blob.generate (normal 0 d)) ~>> biasNet
  where
    d = 1 / sqrt (sqrt (a * b))
    [a, b] = map fromIntegral (dimVal (Proxy @[a,b]))
    run (BTensor ws, xs) = let ys = ws `mulMV` xs
                           in return (ys, backward ws xs)
    backward ws xs dys = let dxs = dys `mulVM` ws
                             dws = dys `outerVV` xs
                         in return (dws, dxs)
