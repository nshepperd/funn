{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
module AI.Funn.Flat.Blob (Blob, generate, split, cat,
                          blob, getBlob, mapBlob,
                          fromList, toList, scale, adamBlob,
                         ) where

import           Control.Applicative
import           Control.DeepSeq
import qualified Data.Binary as LB
import           Data.Foldable hiding (toList)
import qualified Data.Foldable as F
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import           Foreign.C
import           Foreign.Ptr
import           GHC.TypeLits
import qualified Numeric.LinearAlgebra.HMatrix as HM
import           System.IO.Unsafe

import           AI.Funn.Common
import           AI.Funn.SGD
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Space
import           AI.Funn.Flat.Buffer (Buffer)
import qualified AI.Funn.Flat.Buffer as Buffer

newtype Blob (n :: Nat) = Blob Buffer
                        deriving (Show, Read)

blob :: S.Vector Double -> Blob n
blob v = Blob (Buffer.fromVector v)

getBlob :: Blob n -> S.Vector Double
getBlob (Blob buf) = Buffer.getVector buf

instance (Applicative m, KnownNat n) => Zero m (Blob n) where
  zero = pure $ blob (V.replicate n 0)
    where n = natInt (Proxy :: Proxy n)

instance (Applicative m, KnownNat n) => Semi m (Blob n) where
  plus a b = pure $ blob (getBlob a + getBlob b)

instance (Applicative m, KnownNat n) => Additive m (Blob n) where
  plusm blobs = pure $ sumBlobs (F.toList blobs)

instance (Applicative m, KnownNat n) => Scale m Double (Blob n) where
  scale x b = pure $ blob (V.map (x*) (getBlob b))

instance (Applicative m, KnownNat n) => VectorSpace m Double (Blob n) where
  {}

instance (Applicative m, KnownNat n) => Inner m Double (Blob n) where
  inner u v = pure (getBlob u HM.<.> getBlob v)

instance (Applicative m, KnownNat n) => Finite m Double (Blob n) where
  getBasis b = pure (toList b)

instance Eq (Blob n) where
  b1 == b2  =  getBlob b1 == getBlob b2

instance Derivable (Blob n) where
  type D (Blob n) = Blob n

instance NFData (Blob n) where
  rnf (Blob v) = rnf v

instance LB.Binary (Blob n) where
  put xs = putVector putDouble (getBlob xs)
  get = blob <$> getVector getDouble

-- Functions --

generate :: forall f n. (Applicative f, KnownNat n) => f Double -> f (Blob n)
generate f = fromList <$> sequenceA (replicate n f)
  where
    n = natInt (Proxy :: Proxy n)

fromList :: [Double] -> Blob n
fromList xs = blob (V.fromList xs)

toList :: Blob n -> [Double]
toList xs = V.toList (getBlob xs)

split :: forall a b. (KnownNat a, KnownNat b) => Blob (a + b) -> (Blob a, Blob b)
split xs = let v = getBlob xs in
             (blob (V.take s1 v), blob (V.drop s1 v))
  where
    s1 = natInt (Proxy :: Proxy a)

cat :: (KnownNat a, KnownNat b) => Blob a -> Blob b -> Blob (a + b)
cat x@(Blob as) y@(Blob bs)
  | a == 0 = Blob bs
  | b == 0 = Blob as
  | otherwise = Blob (as <> bs)
  where
    a = natVal x
    b = natVal y

-- Special --

natInt :: KnownNat n => proxy n -> Int
natInt p = fromIntegral (natVal p)

sumBlobs :: forall n. (KnownNat n) => [Blob n] -> Blob n
sumBlobs [] = unit
sumBlobs [x] = x
sumBlobs xs = Blob $ Buffer.sumBuffers [buf | Blob buf <- xs]

mapBlob :: (Double -> Double) -> Blob n -> Blob n
mapBlob f b = blob (V.map f (getBlob b))

adamBlob :: forall m (n :: Nat). (Monad m, KnownNat n) => AdamConfig m (Blob n) (Blob n)
adamBlob = defaultAdam {
  adam_pure_d = \x -> generate (pure x),
  adam_scale_d = \x b -> scale x b,
  adam_add_d = plus,
  adam_square_d = \b -> pure $ mapBlob (^2) b,
  adam_sqrt_d = \b -> pure $ mapBlob sqrt b,
  adam_divide_d = \x y -> pure $ blob (V.zipWith (/) (getBlob x) (getBlob y)),
  adam_update_p = plus
  }
  where
    n = fromIntegral (natVal (Proxy :: Proxy n))
