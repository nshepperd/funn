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

newtype BlobT (q :: Mutable) s (n :: Nat) a = Blob (Buffer s a)
type Blob = BlobT I
type MBlob = BlobT M

-- Classes --

instance Derivable (Blob s n a) where
  type D (Blob s n a) = Blob s n a

instance (MonadCL s m, KnownNat n, Floats a) => Zero m (Blob s n a) where
  zero = pureBlob 0

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => Semi m (Blob s n a) where
  plus = addBlob

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => Additive m (Blob s n a) where
  plusm = addBlobs . F.toList

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => Scale m Double (Blob s n a) where
  scale = scaleBlob

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => VectorSpace m Double (Blob s n a) where
  {}

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => Inner m Double (Blob s n a) where
  inner x y = do dot <- mulBlob x y
                 sum <$> toList dot

instance (MonadCL s m, KnownNat n, Floats a, CLNum a) => Finite m Double (Blob s n a) where
  getBasis b = toList b

instance (KnownNat n, Storable a, CLNum a) => CLType (Blob s n a) (ArrayR a) s where
  karg = blobArg

instance (KnownNat n, Storable a, CLNum a) => CLType (MBlob s n a) (ArrayW a) s where
  karg = blobArg

-- Main operations --

freeBlob :: BlobT q s n a -> OpenCL s ()
freeBlob (Blob mem) = Buffer.free mem

createBlob :: forall n m s a. (MonadCL s m, KnownNat n, Storable a) => m (MBlob s n a)
createBlob = Blob <$> Buffer.malloc (fromIntegral n)
  where
    n = natVal (Proxy :: Proxy n)

fromList :: forall n a m s q. (MonadCL s m, KnownNat n, Floats a) => [Double] -> m (BlobT q s n a)
fromList xs = do when (n /= genericLength xs) $ liftIO $ do
                   throwIO (IndexOutOfBounds "Blob.fromList")
                 Blob <$> Buffer.fromList (map fromDouble xs)
  where
    n = natVal (Proxy :: Proxy n)

toList :: (MonadCL s m, Floats a) => BlobT q s n a -> m [Double]
toList (Blob mem) = map toDouble <$> Buffer.toList mem

blobArg :: BlobT q s n a -> KernelArg s
blobArg (Blob mem) = Buffer.arg mem

freeze :: (MonadCL s m, Storable a) => MBlob s n a -> m (Blob s n a)
freeze (Blob mem) = Blob <$> Buffer.clone mem

unsafeFreeze :: (MonadCL s m, Storable a) => MBlob s n a -> m (Blob s n a)
unsafeFreeze (Blob mem) = pure (Blob mem)

thaw :: (MonadCL s m, Storable a) => Blob s n a -> m (MBlob s n a)
thaw (Blob mem) = Blob <$> Buffer.clone mem

unsafeThaw :: (MonadCL s m, Storable a) => Blob s n a -> m (MBlob s n a)
unsafeThaw (Blob mem) = pure (Blob mem)

-- Arithmetic operations --

pureBlob :: forall n a q m s. (MonadCL s m, KnownNat n, Floats a) => Double -> m (BlobT q s n a)
pureBlob x = Blob <$> Buffer.fromVector (S.replicate n (fromDouble x))
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

scaleBlob :: forall n a m s. (MonadCL s m, KnownNat n, Floats a, CLNum a) => Double -> Blob s n a -> m (Blob s n a)
scaleBlob a xs = do ys <- createBlob
                    liftOpenCL (scale [n] a xs ys)
                    unsafeFreeze ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

    scale :: [Int] -> Double -> Blob s n a -> MBlob s n a -> OpenCL s ()
    scale = clfun $ \a xs ys -> do
      i <- get_global_id 0
      at ys i .= castDouble a * at xs i

    castDouble :: Expr Double -> Expr a
    castDouble (Expr e) = Expr e

addBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLNum a) => Blob s n a -> Blob s n a -> m (Blob s n a)
addBlob = zipWithBlob' (+)

addBlobs :: forall n m s a. (MonadCL s m, KnownNat n, Floats a, CLNum a) => [Blob s n a] -> m (Blob s n a)
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

subBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLNum a) => Blob s n a -> Blob s n a -> m (Blob s n a)
subBlob = zipWithBlob' (-)

mulBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLNum a) => Blob s n a -> Blob s n a -> m (Blob s n a)
mulBlob = zipWithBlob' (*)

divideBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLFractional a) => Blob s n a -> Blob s n a -> m (Blob s n a)
divideBlob = zipWithBlob' (/)

squareBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLNum a) => Blob s n a -> m (Blob s n a)
squareBlob = mapBlob' (^2)

sqrtBlob :: forall n m s a. (MonadCL s m, KnownNat n, CLFloating a) => Blob s n a -> m (Blob s n a)
sqrtBlob = mapBlob' sqrt

catBlob :: forall α β m s a q. (MonadCL s m, KnownNat α, KnownNat β, Storable a) => BlobT q s α a -> BlobT q s β a -> m (BlobT q s (α + β) a)
catBlob (Blob xs) (Blob ys) = Blob <$> Buffer.concat [xs, ys]

splitBlob :: forall m n s a q. (KnownNat m, KnownNat n) => BlobT q s (m + n) a -> (BlobT q s m a, BlobT q s n a)
splitBlob (Blob xs) = (Blob ys, Blob zs)
  where
    m = fromIntegral $ natVal (Proxy :: Proxy m)
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    ys = Buffer.slice 0 m xs
    zs = Buffer.slice m n xs

mapBlob :: forall n m s a. (MonadCL s m, KnownNat n, Storable a,
                            Argument (Array W a), Argument (Array R a))
        => (Expr a -> CL (Expr a))
        -> Blob s n a -> m (Blob s n a)
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

zipWithBlob :: forall n m s a. (MonadCL s m, KnownNat n, Storable a,
                                Argument (Array W a), Argument (Array R a))
            => (Expr a -> Expr a -> CL (Expr a))
            -> Blob s n a -> Blob s n a -> m (Blob s n a)
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


mapBlob' :: forall n m s a. (MonadCL s m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a)) => (Expr a -> Expr a) -> Blob s n a -> m (Blob s n a)
mapBlob' f = mapBlob (pure . f)

zipWithBlob' :: forall n m s a. (MonadCL s m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a))
            => (Expr a -> Expr a -> Expr a)
            -> Blob s n a -> Blob s n a -> m (Blob s n a)
zipWithBlob' f = zipWithBlob (\x y -> pure (f x y))


adamBlob :: forall (n :: Nat) m s a. (MonadCL s m, KnownNat n, CLFloating a, Floats a) => AdamConfig m (Blob s n a) (Blob s n a)
adamBlob = defaultAdam {
  adam_pure_d = pureBlob,
  adam_scale_d = scale,
  adam_add_d = plus,
  adam_square_d = squareBlob,
  adam_sqrt_d = sqrtBlob,
  adam_divide_d = divideBlob,
  adam_update_p = plus
  }
