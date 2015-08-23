{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Flat (Blob(..),
                     fcLayer, preluLayer, mergeLayer, sigmoidLayer, splitLayer,
                     quadraticCost, softmaxCost, generateBlob) where

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

import           AI.Funn.Network
import           AI.Funn.Common

newtype Blob (n :: Nat) = Blob { getBlob :: S.Vector Double }
                        deriving (Show, Read)

natInt :: (KnownNat n) => proxy n -> Int
natInt p = fromIntegral (natVal p)

instance (KnownNat n) => VectorSpace (Blob n) where
  (Blob a) ## (Blob b) = Blob (a + b)
  scale v (Blob x) = Blob (HM.scale v x)
  unit = Blob (V.replicate n 0)
    where n = natInt (Proxy :: Proxy n)

instance Derivable (Blob n) where
  type D (Blob n) = Blob n

instance NFData (Blob n) where
  rnf (Blob v) = rnf v

instance CheckNAN (Blob n) where
  check s (Blob xs) b = check s xs b

generateBlob :: forall f n. (Applicative f, KnownNat n) => f Double -> f (Blob n)
generateBlob f = Blob . V.fromList <$> sequenceA (replicate n f)
  where
    n = natInt (Proxy :: Proxy n)

fcLayer :: forall n1 n2 m. (Monad m, KnownNat n1, KnownNat n2) => Network m (Blob n1) (Blob n2)
fcLayer = Network ev numpar initial
  where
    ev (Parameters p) (Blob !input) =
      let w = HM.reshape from (V.slice 0 (from*to) p)
          bs = V.slice (from*to) to p
          output = V.zipWith (+) (w HM.#> input) bs
          !_ = if any isNaN (V.toList output) then
                 error $ "NaN: fcLayer output " ++ show (w, bs, input)
               else ()
          backward (Blob !δ) =
            let da = Blob $ HM.tr w HM.#> δ
                dw = Parameters $ (δ `flat_outer` input)
                db = Parameters $ δ
            in return (da, [dw, db])
      in return (Blob output, 0, backward)
    numpar = from * to + to
    initial = do let σ = sqrt $ 2 / sqrt (fromIntegral (from * to))
                 ws <- V.replicateM (from * to) (normal 0 σ)
                 let (u,_,v) = HM.thinSVD (HM.reshape from ws)
                     m = HM.flatten (u <> HM.tr v) -- orthogonal initialisation
                 bs <- V.replicateM to (pure 0)
                 return $ Parameters (m <> bs)
    from = natInt (Proxy :: Proxy n1)
    to   = natInt (Proxy :: Proxy n2)

prelu, prelu' :: Double -> Double -> Double
prelu α x = max 0 x + α * min 0 x
prelu' α x = if x > 0 then 1 else α

preluLayer :: (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
preluLayer = Network ev 1 (pure $ Parameters $ V.singleton 0.5)
  where
    ev (Parameters p) (Blob !input) =
          let α = V.head p
              output = V.map (prelu α) input
              backward (Blob !δ) = let di = V.zipWith (*) δ (V.map (prelu' α) input)
                                       dα = V.singleton $ V.sum $ V.zipWith (*) δ (V.map (min 0) input)
                                   in return (Blob di, [Parameters dα])
          in return (Blob output, 0, backward)

sigmoidLayer :: (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
sigmoidLayer = Network ev 0 (pure mempty)
  where
    ev _ (Blob !input) =
          let output = V.map σ input
              !_ = if any isNaN (V.toList output) then
                     error $ "NaN: sigmoidLayer output " ++ show (input)
               else ()
              backward (Blob !δ) = let di = V.zipWith (\y δ -> y * (1 - y) * δ) output δ
                                   in return (Blob di, [])
          in return (Blob output, 0, backward)

    σ x = if x < 0 then exp x / (1 + exp x) else 1 / (1 + exp (-x))

splitBlob :: forall a b. (KnownNat a, KnownNat b) => Blob (a + b) -> (Blob a, Blob b)
splitBlob (Blob xs) = (Blob (V.take s1 xs), Blob (V.drop s1 xs))
  where
    s1 = natInt (Proxy :: Proxy a)

concatBlob :: (KnownNat a, KnownNat b) => Blob a -> Blob b -> Blob (a + b)
concatBlob (Blob as) (Blob bs) = Blob (as <> bs)

mergeLayer :: (Monad m, KnownNat a, KnownNat b) => Network m (Blob a, Blob b) (Blob (a + b))
mergeLayer = Network ev 0 (pure mempty)
  where ev _ (!a, !b) =
          let backward δ = return (splitBlob δ, mempty)
          in return (concatBlob a b, 0, backward)

splitLayer :: (Monad m, KnownNat a, KnownNat b) => Network m (Blob (a + b)) (Blob a, Blob b)
splitLayer = Network ev 0 (pure mempty)
  where ev _ ab =
          let backward (da, db) = return (concatBlob da db, mempty)
          in return (splitBlob ab, 0, backward)

ssq :: HM.Vector Double -> Double
ssq xs = V.sum $ V.map (\x -> x*x) xs

quadraticCost :: (Monad m, KnownNat n) => Network m (Blob n, Blob n) ()
quadraticCost = Network ev 0 (pure mempty)
  where ev _ (Blob !o, Blob !target)
          = let diff = V.zipWith (-) o target
                backward () = return ((Blob diff, Blob (V.map negate diff)), mempty)
            in return ((), 0.5 * ssq diff, backward)

softmaxCost :: (Monad m, KnownNat n) => Network m (Blob n, Int) ()
softmaxCost = Network ev 0 (pure mempty)
  where ev _ (Blob !o, !target)
          = let ltotal = log (V.sum . V.map exp $ o)
                cost = (-(o V.! target) + ltotal)
                backward _ = let back = V.imap (\j x -> exp(x - ltotal) - if target == j then 1 else 0) o
                             in return ((Blob back, ()), mempty)
            in return ((), cost, backward)


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
