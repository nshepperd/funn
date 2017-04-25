{-# LANGUAGE TypeFamilies, MultiParamTypeClasses, FlexibleContexts, FlexibleInstances #-}
{-# LANGUAGE KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
module AI.Funn.Flat.Blob (Blob(..), generate, split, cat,
                          fromList, toList, scale, adamBlob
                         ) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable hiding (toList)
import qualified Data.Foldable as F
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
import           AI.Funn.SGD
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Space

newtype Blob (n :: Nat) = Blob { getBlob :: S.Vector Double }
                        deriving (Show, Read)

instance (Applicative m, KnownNat n) => Zero m (Blob n) where
  zero = pure (Blob (V.replicate n 0))
    where n = natInt (Proxy :: Proxy n)

instance (Applicative m, KnownNat n) => Semi m (Blob n) where
  plus (Blob a) (Blob b) = pure $ Blob (a + b)

instance (Applicative m, KnownNat n) => Additive m (Blob n) where
  plusm blobs = pure $ sumBlobs (F.toList blobs)

instance (Applicative m, KnownNat n) => Scale m Double (Blob n) where
  scale x (Blob v) = pure $ Blob (V.map (x*) v)

instance (Applicative m, KnownNat n) => VectorSpace m Double (Blob n) where
  {}

instance (Applicative m, KnownNat n) => Inner m Double (Blob n) where
  inner (Blob u) (Blob v) = pure (u HM.<.> v)


instance Derivable (Blob n) where
  type D (Blob n) = Blob n

instance NFData (Blob n) where
  rnf (Blob v) = rnf v

instance CheckNAN (Blob n) where
  check s (Blob xs) b = check s xs b

-- Functions --

generate :: forall f n. (Applicative f, KnownNat n) => f Double -> f (Blob n)
generate f = Blob . V.fromList <$> sequenceA (replicate n f)
  where
    n = natInt (Proxy :: Proxy n)

fromList :: [Double] -> Blob n
fromList xs = Blob (V.fromList xs)

toList :: Blob n -> [Double]
toList (Blob xs) = V.toList xs

split :: forall a b. (KnownNat a, KnownNat b) => Blob (a + b) -> (Blob a, Blob b)
split (Blob xs) = (Blob (V.take s1 xs), Blob (V.drop s1 xs))
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

natInt :: (KnownNat n) => proxy n -> Int
natInt p = fromIntegral (natVal p)

foreign import ccall "vector_add" ffi_vector_add :: CInt -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE vector_add #-}
vector_add :: M.IOVector Double -> S.Vector Double -> IO ()
vector_add tgt src = do M.unsafeWith tgt $ \tbuf -> do
                          S.unsafeWith src $ \sbuf -> do
                            ffi_vector_add (fromIntegral n) tbuf sbuf
  where
    n = M.length tgt

addBlobsIO :: M.IOVector Double -> [Blob n] -> IO ()
addBlobsIO target ys = go target ys
  where
    go target [] = return ()
    go target (Blob v:vs) = do
      vector_add target v
      go target vs

sumBlobs :: forall n. (KnownNat n) => [Blob n] -> Blob n
sumBlobs [] = Diff.unit
sumBlobs [x] = x
sumBlobs xs = Blob $ unsafePerformIO go
  where
    go = do target <- M.replicate n 0
            addBlobsIO target xs
            V.unsafeFreeze target
    n = natInt (Proxy :: Proxy n)

adamBlob :: forall m (n :: Nat). (Monad m, KnownNat n) => AdamConfig m (Blob n) (Blob n)
adamBlob = defaultAdam {
  adam_pure_d = \x -> generate (pure x),
  adam_scale_d = \x b -> scale x b,
  adam_add_d = plus,
  adam_square_d = \(Blob b) -> pure $ Blob (V.map (^2) b),
  adam_sqrt_d = \(Blob b) -> pure $ Blob (V.map sqrt b),
  adam_divide_d = \(Blob x) (Blob y) -> pure $ Blob (V.zipWith (/) x y),
  adam_update_p = plus
  }
  where
    n = fromIntegral (natVal (Proxy :: Proxy n))
