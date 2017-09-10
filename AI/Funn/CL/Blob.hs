{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
module AI.Funn.CL.Blob (
  BlobT(..), Blob, MBlob,
  createBlob, freeBlob,
  freeze, unsafeFreeze,
  thaw, unsafeThaw,
  fromList, toList,
  blobArg,
  pureBlob, scaleBlob, addBlob, subBlob,
  squareBlob, sqrtBlob, divideBlob,
  catBlob, splitBlob,
  mapBlob, zipWithBlob,
  mapBlob', zipWithBlob',
  adamBlob
  ) where

import           Control.Applicative
import           Control.Monad
import           Data.Foldable hiding (toList)
import qualified Data.Foldable as F
import           Data.List
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Foreign.Storable

import           Control.Exception
import           Control.Monad.IO.Class
import           Data.Proxy
import           GHC.Float
import           GHC.TypeLits

import           AI.Funn.CL.Buffer (Buffer)
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Space
import           AI.Funn.SGD
import           AI.Funn.CL.Code as C
import           AI.Funn.CL.Function

data Mutable = I | M deriving (Show, Eq)

newtype BlobT (q :: Mutable) a (n :: Nat) = Blob (Buffer a)
type Blob = BlobT I
type MBlob = BlobT M

-- Classes --

instance Derivable (Blob a n) where
  type D (Blob a n) = Blob a n

instance (MonadIO m, KnownNat n, Floats a) => Zero m (Blob a n) where
  zero = pureBlob 0

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => Semi m (Blob a n) where
  plus = addBlob

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => Additive m (Blob a n) where
  plusm = addBlobs . F.toList

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => Scale m Double (Blob a n) where
  scale = scaleBlob

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => VectorSpace m Double (Blob a n) where
  {}

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => Inner m Double (Blob a n) where
  inner x y = do dot <- mulBlob x y
                 sum <$> toList dot

instance (MonadIO m, KnownNat n, Floats a, CLNum a) => Finite m Double (Blob a n) where
  getBasis b = toList b

instance (KnownNat n, Storable a, CLNum a) => CLType (Blob a n) (ArrayR a) where
  karg = blobArg

instance (KnownNat n, Storable a, CLNum a) => CLType (MBlob a n) (ArrayW a) where
  karg = blobArg

-- Main operations --

freeBlob :: MonadIO m => BlobT q a n -> m ()
freeBlob (Blob mem) = Buffer.free mem

createBlob :: forall n m a. (MonadIO m, KnownNat n, Storable a) => m (MBlob a n)
createBlob = Blob <$> Buffer.malloc (fromIntegral n)
  where
    n = natVal (Proxy :: Proxy n)

fromList :: forall n a m q. (MonadIO m, KnownNat n, Floats a) => [Double] -> m (BlobT q a n)
fromList xs = do when (n /= genericLength xs) $ liftIO $ do
                   throwIO (IndexOutOfBounds "Blob.fromList")
                 Blob <$> Buffer.fromList (map fromDouble xs)
  where
    n = natVal (Proxy :: Proxy n)

toList :: (MonadIO m, Floats a) => BlobT q a n -> m [Double]
toList (Blob mem) = map toDouble <$> Buffer.toList mem

blobArg :: BlobT q a n -> KernelArg
blobArg (Blob mem) = Buffer.arg mem

freeze :: (MonadIO m, Storable a) => MBlob a n -> m (Blob a n)
freeze (Blob mem) = Blob <$> Buffer.clone mem

unsafeFreeze :: (MonadIO m, Storable a) => MBlob a n -> m (Blob a n)
unsafeFreeze (Blob mem) = pure (Blob mem)

thaw :: (MonadIO m, Storable a) => Blob a n -> m (MBlob a n)
thaw (Blob mem) = Blob <$> Buffer.clone mem

unsafeThaw :: (MonadIO m, Storable a) => Blob a n -> m (MBlob a n)
unsafeThaw (Blob mem) = pure (Blob mem)

-- Arithmetic operations --

pureBlob :: forall n a q m. (MonadIO m, KnownNat n, Floats a) => Double -> m (BlobT q a n)
pureBlob x = Blob <$> Buffer.fromVector (S.replicate n (fromDouble x))
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

scaleBlob :: forall n a m. (MonadIO m, KnownNat n, Floats a, CLNum a) => Double -> Blob a n -> m (Blob a n)
scaleBlob a xs = do ys <- createBlob
                    liftIO (scale [n] a xs ys)
                    unsafeFreeze ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

    scale :: [Int] -> Double -> Blob a n -> MBlob a n -> IO ()
    scale = clfun $ \a xs ys -> do
      i <- get_global_id 0
      at ys i .= castDouble a * at xs i

    castDouble :: Expr Double -> Expr a
    castDouble (Expr e) = Expr e

addBlob :: forall n m a. (MonadIO m, KnownNat n, CLNum a) => Blob a n -> Blob a n -> m (Blob a n)
addBlob = zipWithBlob' (+)

addBlobs :: forall n m a. (MonadIO m, KnownNat n, Floats a, CLNum a) => [Blob a n] -> m (Blob a n)
addBlobs [] = zero
addBlobs (Blob one:xss) = do zs <- Blob <$> Buffer.clone one
                             for_ xss $ \xs -> do
                               runKernel addSource "run" [blobArg xs, blobArg zs] [] [n] [1]
                             return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    add :: C.ArrayR a -> C.ArrayW a -> C.CL ()
    add xs zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at zs i + C.at xs i
    addSource = C.kernel add

subBlob :: forall n m a. (MonadIO m, KnownNat n, CLNum a) => Blob a n -> Blob a n -> m (Blob a n)
subBlob = zipWithBlob' (-)

mulBlob :: forall n m a. (MonadIO m, KnownNat n, CLNum a) => Blob a n -> Blob a n -> m (Blob a n)
mulBlob = zipWithBlob' (*)

divideBlob :: forall n m a. (MonadIO m, KnownNat n, CLFractional a) => Blob a n -> Blob a n -> m (Blob a n)
divideBlob = zipWithBlob' (/)

squareBlob :: forall n m a. (MonadIO m, KnownNat n, CLNum a) => Blob a n -> m (Blob a n)
squareBlob = mapBlob' (^2)

sqrtBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloating a) => Blob a n -> m (Blob a n)
sqrtBlob = mapBlob' sqrt

catBlob :: forall α β m a q. (MonadIO m, KnownNat α, KnownNat β, Storable a) => BlobT q a α -> BlobT q a β -> m (BlobT q a (α + β))
catBlob (Blob xs) (Blob ys) = Blob <$> Buffer.concat [xs, ys]

splitBlob :: forall α β a q. (KnownNat α, KnownNat β) => BlobT q a (α + β) -> (BlobT q a α, BlobT q a β)
splitBlob (Blob xs) = (Blob ys, Blob zs)
  where
    m = fromIntegral $ natVal (Proxy :: Proxy α)
    n = fromIntegral $ natVal (Proxy :: Proxy β)
    ys = Buffer.slice 0 m xs
    zs = Buffer.slice m n xs

mapBlob :: forall n m a. (MonadIO m, KnownNat n, Storable a,
                            Argument (Array W a), Argument (Array R a))
        => (Expr a -> CL (Expr a))
        -> Blob a n -> m (Blob a n)
mapBlob f = go
  where
    go xs = do
      ys <- createBlob
      (runKernel fKernel "run"
       [blobArg xs, blobArg ys]
       [] [fromIntegral n] [])
      unsafeFreeze ys

    fKernel = C.kernel fSrc
    fSrc :: ArrayR a -> ArrayW a -> CL ()
    fSrc xs ys = do i <- get_global_id 0
                    y <- f (at xs i)
                    at ys i .= y

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)

zipWithBlob :: forall n m a. (MonadIO m, KnownNat n, Storable a,
                                Argument (Array W a), Argument (Array R a))
            => (Expr a -> Expr a -> CL (Expr a))
            -> Blob a n -> Blob a n -> m (Blob a n)
zipWithBlob f = go
  where
    go xs ys = do
      zs <- createBlob
      (runKernel fKernel "run"
       [blobArg xs, blobArg ys, blobArg zs]
       [] [fromIntegral n] [])
      unsafeFreeze zs

    fKernel = C.kernel fSrc
    fSrc :: ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    fSrc xs ys zs = do i <- get_global_id 0
                       z <- f (at xs i) (at ys i)
                       at zs i .= z

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)


mapBlob' :: forall n m a. (MonadIO m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a)) => (Expr a -> Expr a) -> Blob a n -> m (Blob a n)
mapBlob' f = mapBlob (pure . f)

zipWithBlob' :: forall n m a. (MonadIO m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a))
            => (Expr a -> Expr a -> Expr a)
            -> Blob a n -> Blob a n -> m (Blob a n)
zipWithBlob' f = zipWithBlob (\x y -> pure (f x y))


adamBlob :: forall (n :: Nat) m a. (MonadIO m, KnownNat n, CLFloating a, Floats a) => AdamConfig m (Blob a n) (Blob a n)
adamBlob = defaultAdam {
  adam_pure_d = pureBlob,
  adam_scale_d = scale,
  adam_add_d = plus,
  adam_square_d = squareBlob,
  adam_sqrt_d = sqrtBlob,
  adam_divide_d = divideBlob,
  adam_update_p = plus
  }
