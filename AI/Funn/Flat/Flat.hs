{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Flat.Flat (fcDiff, preluDiff, reluDiff, sigmoidDiff,
                          mergeDiff, splitDiff, sumDiff, tanhDiff,
                          quadraticCost, softmaxCost) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           GHC.Float

import           Control.DeepSeq
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.SGD
import           AI.Funn.Space

-- Diff --

sumDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Double)
sumDiff = Diff run
  where
    run xs = return (V.sum (getBlob xs), backward)
    n = natInt (Proxy :: Proxy n)
    backward δ = return (blob (V.replicate n δ))

fcDiff :: forall x y m. (Monad m, KnownNat x, KnownNat y) => Diff m (Blob (x*y + y), Blob x) (Blob y)
fcDiff = Diff run
  where
    run (bps, bxs) =
      let ps = getBlob bps
          xs = getBlob bxs
          ws = HM.reshape x (V.slice 0 (x*y) ps)
          bs = V.slice (x*y) y ps
          ys = (ws HM.#> xs) + bs
      in return (blob ys, backward ws xs)

    backward ws xs bδs =
      let δs = getBlob bδs
          dxs = blob $ HM.tr ws HM.#> δs
          dws = (δs `flat_outer` xs)
          dbs = δs
      in return (blob (dws <> dbs), dxs)


    x = natInt (Proxy :: Proxy x)
    y = natInt (Proxy :: Proxy y)

prelu, prelu' :: Double -> Double -> Double
prelu α x = max 0 x + α * min 0 x
prelu' α x = if x > 0 then 1 else α

preluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob 1, Blob n) (Blob n)
preluDiff = Diff run
  where
    run (p, xs) =
      let α = V.head (getBlob p)
          output = Blob.mapBlob (prelu α) xs
      in return (output, backward α xs)

    backward α xs δ =
      let dx = V.zipWith (*) (V.map (prelu' α) (getBlob xs)) (getBlob δ)
          dα = V.sum $ V.zipWith (*) (V.map (min 0) (getBlob xs)) (getBlob δ)
      in return (Blob.fromList [dα], blob dx)

reluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
reluDiff = Diff run
  where
    run xs =
      let output = V.map (prelu 0) (getBlob xs)
      in return (blob output, backward xs)

    backward xs δ =
      let dx = V.zipWith (*) (V.map (prelu' 0) (getBlob xs)) (getBlob δ)
      in return (blob dx)

sigmoidDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
sigmoidDiff = Diff run
  where
    run input =
      let output = V.map σ (getBlob input)
      in return (blob output, backward output)

    backward output δs =
      let di = V.zipWith (\y δ -> y * (1 - y) * δ) output (getBlob δs)
      in return (blob di)

    σ x = if x < 0 then
            exp x / (1 + exp x)
          else
            1 / (1 + exp (-x))


tanhDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
tanhDiff = Diff run
  where
    run input =
          let output = V.map tanh (getBlob input)
              backward δs =
                let di = V.zipWith (\y δ -> tanh' y * δ) output (getBlob δs)
                in return (blob di)
          in return (blob output, backward)

    tanh x = (exp x - exp (-x)) / (exp x + exp (-x))
    tanh' y = 1 - y^2

mergeDiff :: (Monad m, KnownNat a, KnownNat b) => Diff m (Blob a, Blob b) (Blob (a + b))
mergeDiff = Diff run
  where run (!a, !b) =
          let backward δ = pure (Blob.split δ)
          in pure (Blob.cat a b, backward)

splitDiff :: (Monad m, KnownNat a, KnownNat b) => Diff m (Blob (a + b)) (Blob a, Blob b)
splitDiff = Diff run
  where run ab =
          let backward (da, db) = pure (Blob.cat da db)
          in pure (Blob.split ab, backward)


quadraticCost :: (Monad m, KnownNat n) => Diff m (Blob n, Blob n) Double
quadraticCost = Diff run
  where
    run (o, target)
      = let os = getBlob o
            ts = getBlob target
            diff = V.zipWith (-) os ts
        in return (0.5 * ssq diff, backward diff)

    backward diff d = do one <- scale d (blob diff)
                         two <- scale (-1) one
                         return (one, two)

    ssq :: HM.Vector Double -> Double
    ssq xs = V.sum $ V.map (\x -> x*x) xs

softmaxCost :: (Monad m, KnownNat n) => Diff m (Blob n, Int) Double
softmaxCost = Diff run
  where run (bo, !target)
          = let os = getBlob bo
                xt = os V.! target
                exp_total_minus_xt = V.imap (\j x -> if j /= target then exp (x - xt) else 0) os
                log_total_minus_xt = log1p (V.sum exp_total_minus_xt)
                cost = log_total_minus_xt
            in return (cost, backward target os)

        backward target os dcost =
          let
            total_but_target = V.sum (V.imap (\i x -> if i /= target then exp x else 0) os)
            total = V.sum (V.map exp os)
            back = V.imap (\j x -> dcost * (if target == j then
                                              -total_but_target / total
                                             else
                                               exp x / total)) os
          in return (blob back, ())

-- Special --

natInt :: (KnownNat n) => proxy n -> Int
natInt p = fromIntegral (natVal p)

foreign import ccall "outer_product" outer_product :: CInt -> CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE flat_outer #-}
flat_outer :: S.Vector Double -> S.Vector Double -> S.Vector Double
flat_outer u v = unsafePerformIO go
  where
    go = do target <- M.new (n*m) :: IO (M.IOVector Double)
            S.unsafeWith u $ \ubuf -> do
              S.unsafeWith v $ \vbuf -> do
                M.unsafeWith target $ \tbuf -> do
                  outer_product (fromIntegral n) (fromIntegral m) ubuf vbuf tbuf
            V.unsafeFreeze target
    n = V.length u
    m = V.length v
