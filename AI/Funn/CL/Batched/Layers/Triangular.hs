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
module AI.Funn.CL.Batched.Layers.Triangular where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.List.Split
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe
import           Text.Printf

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.GLOW (Invertible(..), invert)
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Network (Network(..), liftDiff, network, runNetwork)
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
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space
import           AI.Funn.TypeLits

{-# NOINLINE mulUVProgram #-}
mulUVProgram :: KernelProgram '[TensorCL '[n, n], TensorCL '[ω, n], MTensorCL '[ω, n]]
mulUVProgram = compile $ \ws xs ys -> do
  [u, j] <- traverse get_global_id [0,1]
  let [_, n] = dimsOf xs
  -- ys[j] = sum_{i=j..n-1} ws![j,i] * xs![i]
  acc <- eval $ 1 * xs![u, j]
  forEach (j+1) n $ \i -> do
    acc .= acc + ws![j,i] * xs![u, i]
  ys![u, j] .= acc

-- Computes U * v
mulUV :: KnownDimsF [ω, n] => Tensor [n,n] -> Tensor [ω, n] -> Tensor [ω, n]
mulUV ws ys = unsafePerformIO $ do
  xs <- Tensor.new
  clfun mulUVProgram (dimVal xs) ws ys xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE mulVUProgram #-}
mulVUProgram :: KernelProgram '[TensorCL '[n, n], TensorCL '[ω, n], MTensorCL '[ω, n]]
mulVUProgram = compile $ \ws xs ys -> do
  [u, j] <- traverse get_global_id [0,1]
  let [_, n] = dimsOf xs
  -- ys[j] = sum_{i ≤ j} ws![i,j] * xs![i]
  acc <- eval $ 1 * xs![u, j]
  forEach 0 j $ \i -> do
    acc .= acc + ws![i,j] * xs![u, i]
  ys![u, j] .= acc

-- Computes v' * U
mulVU :: KnownDimsF [ω, n] => Tensor [ω, n] -> Tensor [n,n] -> Tensor [ω, n]
mulVU ys ws = unsafePerformIO $ do
  xs <- Tensor.new
  clfun mulVUProgram (dimVal xs) ws ys xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE invUVProgram #-}
invUVProgram :: KernelProgram '[TensorCL '[n, n], TensorCL '[ω, n], MTensorCL '[ω, n]]
invUVProgram = compile $ \ws ys xs -> do
  [u] <- traverse get_global_id [0]
  let [_, n] = dimsOf xs
  forEach 0 n $ \neg_j -> do
    -- j <- [n-1 .. 0]
    j <- eval (n - 1 - neg_j)
    -- ys[j] = sum_{i=j..n-1} ws![j,i] * xs![i]
    -- ys[j] = ws![j,j] * xs![j] + sum_{i=j+1 .. n-1} ws![j,i] * xs![i]
    -- ws![j,j] * xs![j] = ys[j] - sum_{i=j+1 .. n-1} ws![j,i] * xs![i]
    acc <- eval (ys![u, j])
    forEach (j+1) n $ \i -> do
      acc .= acc - ws![j,i] * xs![u,i]
    xs![u,j] .= acc / 1

-- Computes U^-1 * v
invUV :: KnownDimsF [ω, n] => Tensor [n,n] -> Tensor [ω, n] -> Tensor [ω, n]
invUV ws ys = unsafePerformIO $ do
  xs <- Tensor.new
  clfun invUVProgram (take 1 $ dimVal xs) ws ys xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE invVUProgram #-}
invVUProgram :: KernelProgram '[TensorCL '[n, n], TensorCL '[ω, n], MTensorCL '[ω, n]]
invVUProgram = compile $ \ws ys xs -> do
  [u] <- traverse get_global_id [0]
  let [_, n] = dimsOf xs
  forEach 0 n $ \j -> do
    acc <- eval (ys![u, j])
    forEach 0 j $ \i -> do
      acc .= acc - ws![i,j] * xs![u,i]
    xs![u,j] .= acc / 1

-- Computes v' * U^-1
invVU :: KnownDimsF [ω, n] => Tensor [ω, n] -> Tensor [n,n] -> Tensor [ω, n]
invVU ys ws = unsafePerformIO $ do
  xs <- Tensor.new
  clfun invVUProgram (take 1 $ dimVal xs) ws ys xs :: IO ()
  return (Tensor.unsafeFreeze xs)

{-# NOINLINE outerWWProgram #-}
outerWWProgram :: KernelProgram '[TensorCL '[ω, a], TensorCL '[ω, b], MTensorCL '[ω, a, b]]
outerWWProgram = compile $ \xs ys ws -> do
  [u, i, j] <- traverse get_global_id [0, 1, 2]
  ws![u,i,j] .= xs![u,i] * ys![u,j]

outerWW :: KnownDimsF [ω, a, b] => Tensor [ω, a] -> Tensor [ω, b] -> Tensor [ω, a, b]
outerWW xs ys = unsafePerformIO $ do
  ws <- Tensor.new
  clfun outerWWProgram (dimVal ws) xs ys ws :: IO ()
  return (Tensor.unsafeFreeze ws)

{-# NOINLINE makeUpperProgram #-}
makeUpperProgram :: KernelProgram '[TensorCL '[ω, n, n], MTensorCL '[ω, n, n]]
makeUpperProgram = compile $ \ws us -> do
  [u,i,j] <- traverse get_global_id [0,1,2]
  us![u,i,j] .= fstep (castExpr i + 0.5) (castExpr j + 0.0) * ws![u,i,j]

-- Zeros out entries of ws to make the result strictly upper triangular.
makeUpper :: KnownDimsF [ω,n] => Tensor [ω,n,n] -> Tensor [ω,n,n]
makeUpper ws = unsafePerformIO $ do
  us <- Tensor.new
  clfun makeUpperProgram (dimVal us) ws us :: IO ()
  return (Tensor.unsafeFreeze us)

upperMultiplyNet :: forall n ω m. (MonadIO m, KnownDimsF [ω,n]) => Network m ω _ (Tensor [ω, n]) (Tensor [ω, n])
upperMultiplyNet = network (Diff run) init
  where
    run (BTensor ws :: BTensor ω [n,n], xs) = do
      let ys = ws `mulUV` xs
      return (ys, backward ws xs)
    backward ws xs dys = do
      let dxs = dys `mulVU` ws
          dws = dys `outerWW` xs
      return (makeUpper dws, dxs)

    [n] = dimVal (Proxy @'[n])
    σ = 1 / sqrt (fromIntegral n)
    gen i j
      | i == j = pure 1
      | i < j = normal 0 σ
      | i > j = pure 0
    init = Blob.fromList <$> sequence [gen i j | i <- [0..n-1], j <- [0..n-1]]


upperMultiplyNetInv :: forall n ω m. (MonadIO m, KnownDimsF [ω,n]) => Network m ω _ (Tensor [ω, n]) (Tensor [ω, n])
upperMultiplyNetInv = network (Diff run) init
  where
    run (BTensor ws :: BTensor ω [n,n], xs) = do
      let ys = ws `invUV` xs
      return (ys, backward ws ys)
    backward ws ys dys = do
      let dxs = dys `invVU` ws
          dws = dxs `outerWW` ys
      dws' <- scale (-1) dws
      return (makeUpper dws', dxs)

    [n] = dimVal (Proxy @'[n])
    σ = 1 / sqrt (fromIntegral n)
    gen i j
      | i == j = pure 1
      | i < j = normal 0 σ
      | i > j = pure 0
    init = Blob.fromList <$> sequence [gen i j | i <- [0..n-1], j <- [0..n-1]]

upperMultiplyInv :: forall n ω m. (MonadIO m, KnownDimsF [ω,n]) => Invertible m ω _ (Tensor [ω, n]) (Tensor [ω, n])
upperMultiplyInv = Invertible upperMultiplyNet upperMultiplyNetInv

-- instance KnownDims ds => Show (Tensor ds) where
--   show t = case dimVal (Proxy @ ds) of
--     [h, w] -> let ls = chunksOf w xs in
--       "[" ++ intercalate "\n " [show l | l <- ls] ++ "]"
--     _ -> show xs
--     where
--       xs = unsafePerformIO (Tensor.toList t)
