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
import           AI.Funn.Flat.Blob (Blob(..))
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.SGD
import           AI.Funn.Space

-- Diff --

sumDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Double)
sumDiff = Diff run
  where
    run (Blob !xs) = return (V.sum xs, backward)
    n = natInt (Proxy :: Proxy n)
    backward δ = return (Blob (V.replicate n δ))

fcDiff :: forall x y m. (Monad m, KnownNat x, KnownNat y) => Diff m (Blob (x*y + y), Blob x) (Blob y)
fcDiff = Diff run
  where
    run (Blob ps, Blob xs) =
      let ws = HM.reshape x (V.slice 0 (x*y) ps)
          bs = V.slice (x*y) y ps
          ys = (ws HM.#> xs) + bs
          backward (Blob δs) =
            let dxs = Blob $ HM.tr ws HM.#> δs
                dws = (δs `flat_outer` xs)
                dbs = δs
            in return (Blob (dws <> dbs), dxs)
      in return (Blob ys, backward)
    x = natInt (Proxy :: Proxy x)
    y = natInt (Proxy :: Proxy y)

prelu, prelu' :: Double -> Double -> Double
prelu α x = max 0 x + α * min 0 x
prelu' α x = if x > 0 then 1 else α

preluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob 1, Blob n) (Blob n)
preluDiff = Diff run
  where
    run (Blob p, Blob !xs) =
      let α = V.head p
          output = V.map (prelu α) xs
          backward (Blob !δ) = let dx = V.zipWith (*) δ (V.map (prelu' α) xs)
                                   dα = V.sum $ V.zipWith (*) δ (V.map (min 0) xs)
                               in return (Blob.fromList [dα], Blob dx)
      in return (Blob output, backward)

reluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
reluDiff = Diff run
  where
    run (Blob !xs) =
      let α = 0
          output = V.map (prelu α) xs
          backward (Blob !δ) = let dx = V.zipWith (*) δ (V.map (prelu' α) xs)
                               in return (Blob dx)
      in return (Blob output, backward)

sigmoidDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
sigmoidDiff = Diff run
  where
    run (Blob !input) =
          let output = V.map σ input
              backward (Blob !δs) =
                let di = V.zipWith (\y δ -> y * (1 - y) * δ) output δs
                in return (Blob di)
          in return (Blob output, backward)

    σ x = if x < 0 then
            exp x / (1 + exp x)
          else
            1 / (1 + exp (-x))


tanhDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
tanhDiff = Diff run
  where
    run (Blob !input) =
          let output = V.map tanh input
              backward (Blob !δs) =
                let di = V.zipWith (\y δ -> tanh' y * δ) output δs
                in return (Blob di)
          in return (Blob output, backward)

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
    run (Blob !o, Blob !target)
      = let diff = V.zipWith (-) o target
            backward dcost = do one <- scale dcost (Blob diff)
                                two <- scale (-1) one
                                return (one, two)
        in return (0.5 * ssq diff, backward)

    ssq :: HM.Vector Double -> Double
    ssq xs = V.sum $ V.map (\x -> x*x) xs

softmaxCost :: (Monad m, KnownNat n) => Diff m (Blob n, Int) Double
softmaxCost = Diff run
  where run (Blob !o, !target)
          = let ltotal = log (V.sum . V.map exp $ o)
                cost = (-(o V.! target) + ltotal)
                backward dcost = let back = V.imap (\j x -> exp(x - ltotal) - if target == j then dcost else 0) o
                                 in return (Blob back, ())
            in return (cost, backward)

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

adamBlob :: forall m (n :: Nat). (Monad m, KnownNat n) => AdamConfig m (Blob n) (Blob n)
adamBlob = defaultAdam {
  adam_pure_d = \x -> Blob.generate (pure x),
  adam_scale_d = \x b -> scale x b,
  adam_add_d = plus,
  adam_square_d = \(Blob b) -> pure $ Blob (V.map (^2) b),
  adam_sqrt_d = \(Blob b) -> pure $ Blob (V.map sqrt b),
  adam_divide_d = \(Blob x) (Blob y) -> pure $ Blob (V.zipWith (/) x y),
  adam_update_p = plus
  }
  where
    n = fromIntegral (natVal (Proxy :: Proxy n))
