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
module AI.Funn.CL.Layers.FullyConnected (mulMV, mulVM, outerVV, fcNet) where

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

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.Layers.Tensor
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor(..), MTensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits

{-# NOINLINE mulMVProgram #-}
mulMVProgram :: KernelProgram '[TensorCL '[b, a], TensorCL '[a], MTensorCL '[b]]
mulMVProgram = compile $ \ws xs ys -> do
  j <- get_global_id 0
  let [a] = dimsOf xs
  acc <- eval 0
  forEach 0 a $ \i -> do
    acc .= acc + ws![j,i] * xs![i]
  ys![j] .= acc

mulMV :: KnownDimsF [a, b] => Tensor [b,a] -> Tensor '[a] -> Tensor '[b]
mulMV ws xs = unsafePerformIO $ do
  ys <- Tensor.new
  clfun mulMVProgram (dimVal ys) ws xs ys :: IO ()
  return (Tensor.unsafeFreeze ys)

{-# NOINLINE mulVMProgram #-}
mulVMProgram :: KernelProgram '[TensorCL '[b], TensorCL '[b, a], MTensorCL '[a]]
mulVMProgram = compile $ \ys ws xs -> do
  i <- get_global_id 0
  let [b] = dimsOf ys
  acc <- eval 0
  forEach 0 b $ \j -> do
    acc .= acc + ws![j,i] * ys![j]
  xs![i] .= acc

mulVM :: KnownDimsF [a, b] => Tensor '[b] -> Tensor [b,a] -> Tensor '[a]
mulVM ys ws = unsafePerformIO $ do
  xs <- Tensor.new
  clfun mulVMProgram (dimVal xs) ys ws xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE outerVVProgram #-}
outerVVProgram :: KernelProgram '[TensorCL '[a], TensorCL '[b], MTensorCL '[a, b]]
outerVVProgram = compile $ \xs ys ws -> do
  ~[i,j] <- traverse get_global_id [0,1]
  ws![i,j] .= xs![i] * ys![j]

outerVV :: KnownDimsF [a, b] => Tensor '[a] -> Tensor '[b] -> Tensor [a,b]
outerVV xs ys = unsafePerformIO $ do
  ws <- Tensor.new
  clfun outerVVProgram (dimVal ws) xs ys ws :: IO ()
  return (Tensor.unsafeFreeze ws)

fcNet :: forall a b m. (MonadIO m, KnownDimsF [a,b])
      => Network m _ (Tensor '[a]) (Tensor '[b])
fcNet = network (Diff run) (Blob.generate (normal 0 d)) ~>> biasNet
  where
    d = 1 / sqrt (sqrt (a * b))
    [a, b] = map fromIntegral (dimVal (Proxy @[a,b]))
    run (ws, xs) = let ys = ws `mulMV` xs
                   in return (ys, backward ws xs)
    backward ws xs dys = let dxs = dys `mulVM` ws
                             dws = dys `outerVV` xs
                         in return (dws, dxs)
