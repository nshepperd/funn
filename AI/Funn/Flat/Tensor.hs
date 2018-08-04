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

module AI.Funn.Flat.Tensor (
  Tensor(..),
  -- Core
  fromBlob,
  fromList,
  fromVector,
  replicate,
  replicateM,
  toBlob,
  toList,
  toVector,
  -- Arithmetic
  reshape,
  mapTensor,
  zipWithTensor
  ) where

import           Control.Applicative
import           Control.DeepSeq
import qualified Data.Binary as LB
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           GHC.TypeLits
import           Prelude hiding (replicate)
import           System.IO.Unsafe

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob(..))
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Flat.Buffer (Buffer)
import qualified AI.Funn.Flat.Buffer as Buffer
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space

newtype Tensor (ds :: [Nat]) = Tensor Buffer
  deriving (Show, Read)

instance Eq (Tensor ds) where
  b1 == b2  = toVector b1 == toVector b2

instance Derivable (Tensor ds) where
  type D (Tensor ds) = Tensor ds

instance (Applicative m, KnownDims ds) => Zero m (Tensor ds) where
  zero = pure $ fromVector (V.replicate n 0)
    where n = dimSize (Proxy @ ds)

instance (Applicative m, KnownDims ds) => Semi m (Tensor ds) where
  plus a b = pure $ fromVector (V.zipWith (+) (toVector a) (toVector b))

instance (Applicative m, KnownDims ds) => Additive m (Tensor ds) where
  plusm ts = pure $ Tensor $ Buffer.sumBuffers [buf | Tensor buf <- F.toList ts]

instance (Applicative m, KnownDims ds) => Scale m Double (Tensor ds) where
  scale x t = pure $ fromVector (V.map (x*) (toVector t))

instance (Applicative m, KnownDims ds) => VectorSpace m Double (Tensor ds) where
  {}

instance (Applicative m, KnownDims ds) => Inner m Double (Tensor ds) where
  inner u v = pure (V.sum $ V.zipWith (*) (toVector u) (toVector v))

instance (Applicative m, KnownDims ds) => Finite m Double (Tensor ds) where
  getBasis b = pure (toList b)



instance (Monad m, KnownDims ds) => AdamOps m (Tensor ds) where
  adam_pure_d x = pure (replicate x)
  adam_square_d b = pure (mapTensor (^2) b)
  adam_sqrt_d b = pure (mapTensor sqrt b)
  adam_divide_d x y = pure (zipWithTensor (/) x y)

instance (Monad m, KnownDims ds) => Adam m (Tensor ds) (Tensor ds) where
  adam_update_p = plus


fromVector :: KnownDims ds => S.Vector Double -> Tensor ds
fromVector xs
  | V.length xs == n = out
  | otherwise = error "Size mismatch in Tensor.fromList"
  where
    n = dimSize out
    out = Tensor (Buffer.fromVector xs)

toVector :: Tensor ds -> S.Vector Double
toVector (Tensor buf) = Buffer.getVector buf

fromList :: KnownDims ds => [Double] -> Tensor ds
fromList xs = fromVector (V.fromList xs)

toList :: Tensor ds -> [Double]
toList xs = V.toList (toVector xs)

fromBlob :: Blob (Prod ds) -> Tensor ds
fromBlob (Blob buf) = Tensor buf

toBlob :: Tensor ds -> Blob (Prod ds)
toBlob (Tensor buf) = Blob buf

replicate :: forall ds. KnownDims ds => Double -> Tensor ds
replicate x = Tensor (Buffer.fromVector (V.replicate n x))
  where
    n = dimSize (Proxy @ ds)

replicateM :: forall ds m. (KnownDims ds, Monad m) => m Double -> m (Tensor ds)
replicateM x = Tensor . Buffer.fromVector <$> (V.replicateM n x)
  where
    n = dimSize (Proxy @ ds)

reshape :: (Prod as ~ Prod bs) => Tensor as -> Tensor bs
reshape (Tensor buf) = Tensor buf


mapTensor :: (Double -> Double) -> Tensor ds -> Tensor ds
mapTensor f (Tensor buf) = Tensor (Buffer.fromVector $ V.map f $ Buffer.getVector buf)

zipWithTensor :: (Double -> Double -> Double) -> Tensor ds -> Tensor ds -> Tensor ds
zipWithTensor f (Tensor b1) (Tensor b2) = Tensor (Buffer.fromVector $ V.zipWith f v1 v2)
  where
    v1 = Buffer.getVector b1
    v2 = Buffer.getVector b2
