{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, TypeApplications #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Network.Flat (sumLayer, fcLayer,
                             preluLayer, reluLayer, sigmoidLayer,
                             mergeLayer, splitLayer, tanhLayer,
                             quadraticCost, softmaxCost
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
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob

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
                 return $ blob (m <> bs)
    from = fromIntegral $ natVal (Proxy @ x)
    to = fromIntegral $ natVal (Proxy @ y)

preluLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
preluLayer = Network Proxy Flat.preluDiff initial
  where
    -- Looks Linear (LL) initialisation
    initial = Blob.generate (pure 1)

dupLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n, Blob n)
dupLayer = liftDiff Diff.dup

reluLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
reluLayer = liftDiff Flat.reluDiff

sigmoidLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
sigmoidLayer = liftDiff Flat.sigmoidDiff

tanhLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n) (Blob n)
tanhLayer = liftDiff Flat.tanhDiff

mergeLayer :: (Monad m, KnownNat a, KnownNat b) => Network m (Blob a, Blob b) (Blob (a + b))
mergeLayer = liftDiff Flat.mergeDiff

splitLayer :: (Monad m, KnownNat a, KnownNat b) => Network m (Blob (a + b)) (Blob a, Blob b)
splitLayer = liftDiff Flat.splitDiff

quadraticCost :: (Monad m, KnownNat n) => Network m (Blob n, Blob n) Double
quadraticCost = liftDiff Flat.quadraticCost

softmaxCost :: (Monad m, KnownNat n) => Network m (Blob n, Int) Double
softmaxCost = liftDiff Flat.softmaxCost
