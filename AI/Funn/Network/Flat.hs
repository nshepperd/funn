{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, TypeApplications #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Network.Flat (sumLayer, fcLayer
                             -- preluLayer, reluLayer, sigmoidLayer,
                             -- mergeLayer, splitLayer, tanhLayer,
                             -- quadraticCost, softmaxCost
                            ) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random

import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import qualified Numeric.LinearAlgebra.HMatrix as HM
import           Control.DeepSeq

import           AI.Funn.Common
import           AI.Funn.Network.Network
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Flat as Flat
import           AI.Funn.Flat.Flat (Blob(..))

-- Diff --

sumLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Double)
sumLayer = liftDiff Flat.sumDiff

fcLayer :: forall x y m. (Monad m, KnownNat x, KnownNat y) => Network m (Blob x) (Blob y)
fcLayer = Network Proxy Flat.fcDiff initial
  where
    initial = do let σ = sqrt $ 2 / sqrt (fromIntegral (from * to))
                 ws <- V.replicateM (from * to) (normal 0 σ)
                 let (u,_,v) = HM.thinSVD (HM.reshape from ws)
                     m = HM.flatten (u <> HM.tr v) -- orthogonal initialisation
                 bs <- V.replicateM to (pure 0)
                 return $ Blob (m <> bs)
    from = fromIntegral $ natVal (Proxy @ x)
    to = fromIntegral $ natVal (Proxy @ y)

-- preluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob 1, Blob n) (Blob n)
-- preluDiff = Diff run
--   where
--     run (Blob p, Blob !xs) =
--       let α = V.head p
--           output = V.map (prelu α) xs
--           backward (Blob !δ) = let dx = V.zipWith (*) δ (V.map (prelu' α) xs)
--                                    dα = V.sum $ V.zipWith (*) δ (V.map (min 0) xs)
--                                in return (unsafeBlob [dα], Blob dx)
--       in return (Blob output, backward)

-- reluDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
-- reluDiff = Diff run
--   where
--     run (Blob !xs) =
--       let α = 0
--           output = V.map (prelu α) xs
--           backward (Blob !δ) = let dx = V.zipWith (*) δ (V.map (prelu' α) xs)
--                                in return (Blob dx)
--       in return (Blob output, backward)

-- sigmoidDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
-- sigmoidDiff = Diff run
--   where
--     run (Blob !input) =
--           let output = V.map σ input
--               backward (Blob !δs) =
--                 let di = V.zipWith (\y δ -> y * (1 - y) * δ) output δs
--                 in return (Blob di)
--           in return (Blob output, backward)

--     σ x = if x < 0 then
--             exp x / (1 + exp x)
--           else
--             1 / (1 + exp (-x))


-- tanhDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n) (Blob n)
-- tanhDiff = Diff run
--   where
--     run (Blob !input) =
--           let output = V.map tanh input
--               backward (Blob !δs) =
--                 let di = V.zipWith (\y δ -> tanh' y * δ) output δs
--                 in return (Blob di)
--           in return (Blob output, backward)

--     tanh x = (exp x - exp (-x)) / (exp x + exp (-x))
--     tanh' y = 1 - y^2

-- mergeDiff :: (Monad m, KnownNat a, KnownNat b) => Diff m (Blob a, Blob b) (Blob (a + b))
-- mergeDiff = Diff run
--   where run (!a, !b) =
--           let backward δ = pure (splitBlob δ)
--           in pure (concatBlob a b, backward)

-- splitDiff :: (Monad m, KnownNat a, KnownNat b) => Diff m (Blob (a + b)) (Blob a, Blob b)
-- splitDiff = Diff run
--   where run ab =
--           let backward (da, db) = pure (concatBlob da db)
--           in pure (splitBlob ab, backward)


-- quadraticCost :: (Monad m, KnownNat n) => Diff m (Blob n, Blob n) Double
-- quadraticCost = Diff run
--   where
--     run (Blob !o, Blob !target)
--       = let diff = V.zipWith (-) o target
--             backward dcost = return ((scaleBlob dcost (Blob diff),
--                                       scaleBlob dcost (Blob (V.map negate diff))))
--         in return (0.5 * ssq diff, backward)

--     ssq :: HM.Vector Double -> Double
--     ssq xs = V.sum $ V.map (\x -> x*x) xs

-- softmaxCost :: (Monad m, KnownNat n) => Diff m (Blob n, Int) Double
-- softmaxCost = Diff run
--   where run (Blob !o, !target)
--           = let ltotal = log (V.sum . V.map exp $ o)
--                 cost = (-(o V.! target) + ltotal)
--                 backward dcost = let back = V.imap (\j x -> exp(x - ltotal) - if target == j then dcost else 0) o
--                                  in return (Blob back, ())
--             in return (cost, backward)

-- -- Special --

-- natInt :: (KnownNat n) => proxy n -> Int
-- natInt p = fromIntegral (natVal p)

-- foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

-- {-# NOINLINE vector_add #-}
-- vector_add :: M.IOVector Double -> S.Vector Double -> IO ()
-- vector_add tgt src = do M.unsafeWith tgt $ \tbuf -> do
--                           S.unsafeWith src $ \sbuf -> do
--                             ffi_vector_add (fromIntegral n) tbuf sbuf
--   where
--     n = M.length tgt

-- addBlobsIO :: M.IOVector Double -> [Blob n] -> IO ()
-- addBlobsIO target ys = go target ys
--   where
--     go target [] = return ()
--     go target (Blob v:vs) = do
--       vector_add target v
--       go target vs

-- sumBlobs :: forall n. (KnownNat n) => [Blob n] -> Blob n
-- sumBlobs [] = Diff.unit
-- sumBlobs [x] = x
-- sumBlobs xs = Blob $ unsafePerformIO go
--   where
--     go = do target <- M.replicate n 0
--             addBlobsIO target xs
--             V.unsafeFreeze target
--     n = natInt (Proxy :: Proxy n)

-- foreign import ccall "outer_product" outer_product :: CInt -> CInt -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

-- {-# NOINLINE flat_outer #-}
-- flat_outer :: S.Vector Double -> S.Vector Double -> S.Vector Double
-- flat_outer u v = unsafePerformIO go
--   where
--     go = do target <- M.new (n*m) :: IO (M.IOVector Double)
--             S.unsafeWith u $ \ubuf -> do
--               S.unsafeWith v $ \vbuf -> do
--                 M.unsafeWith target $ \tbuf -> do
--                   outer_product (fromIntegral n) (fromIntegral m) ubuf vbuf tbuf
--             V.unsafeFreeze target
--     n = V.length u
--     m = V.length v
