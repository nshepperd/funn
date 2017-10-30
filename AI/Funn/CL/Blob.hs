{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
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
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.IORef
import           Data.List
import           Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Foreign.Storable
import           GHC.Float
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Buffer (Buffer)
import qualified AI.Funn.CL.Buffer as Buffer
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.SGD
import           AI.Funn.Space
import qualified Foreign.OpenCL.Bindings as CL

data Mutable = I | M deriving (Show, Eq)

newtype BlobT (q :: Mutable) a (n :: Nat) = Blob (Buffer a)
type Blob = BlobT I
type MBlob = BlobT M

-- Classes --

instance Derivable (Blob a n) where
  type D (Blob a n) = Blob a n

instance (MonadIO m, KnownNat n, Floats a) => Zero m (Blob a n) where
  zero = pureBlob 0

instance (MonadIO m, KnownNat n, CLFloats a) => Semi m (Blob a n) where
  plus = addBlob

instance (MonadIO m, KnownNat n, CLFloats a) => Additive m (Blob a n) where
  plusm = addBlobs . F.toList

instance (MonadIO m, KnownNat n, CLFloats a) => Scale m Double (Blob a n) where
  scale = scaleBlob

instance (MonadIO m, KnownNat n, CLFloats a) => VectorSpace m Double (Blob a n) where
  {}

instance (MonadIO m, KnownNat n, CLFloats a) => Inner m Double (Blob a n) where
  inner x y = do dot <- mulBlob x y
                 sum <$> toList dot

instance (MonadIO m, KnownNat n, CLFloats a) => Finite m Double (Blob a n) where
  getBasis b = toList b

instance (KnownNat n, Storable a, Argument (Array R a)) => CLType (Blob a n) (ArrayR a) where
  karg = blobArg

instance (KnownNat n, Storable a, Argument (Array W a)) => CLType (MBlob a n) (ArrayW a) where
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

createCopy :: forall n a m. (MonadIO m, Storable a, KnownNat n) => Blob a n -> m (MBlob a n)
createCopy src = do dst <- createBlob
                    copy src dst
                    return dst

copy :: forall n a m. (MonadIO m, Storable a, KnownNat n) => Blob a n -> MBlob a n -> m ()
copy (Blob src) (Blob dst) = Buffer.copy src dst 0 0 len
  where
    len = fromIntegral $ natVal (Proxy :: Proxy n)


-- Arithmetic operations --

pureBlob :: forall n a q m. (MonadIO m, KnownNat n, Floats a) => Double -> m (BlobT q a n)
pureBlob x = Blob <$> Buffer.fromVector (S.replicate n (fromDouble x))
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

data KName = Scale Precision
           | AddInto Precision
           | Add Precision
           | Sub Precision
           | Mul Precision
           | Div Precision
           | Square Precision
           | Sqrt Precision
  deriving (Show, Eq, Ord)

{-# NOINLINE memoTable #-}
memoTable :: KTable KName
memoTable = newKTable unsafePerformIO

scaleBlob :: forall n a m. (MonadIO m, KnownNat n, CLFloats a) => Double -> Blob a n -> m (Blob a n)
scaleBlob a xs = do ys <- createBlob
                    liftIO (scale [n] a xs ys)
                    unsafeFreeze ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

    scale :: [Int] -> Double -> Blob a n -> MBlob a n -> IO ()
    scale = memoc memoTable (Scale (precision @a)) $ \a xs ys -> do
      i <- get_global_id 0
      at ys i .= castDouble a * at xs i

    castDouble :: Expr Double -> Expr a
    castDouble (Expr e) = Expr e

addBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> Blob a n -> m (Blob a n)
addBlob = zipWithBlob' memoTable (Add (precision @a)) (+)

addBlobs :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => [Blob a n] -> m (Blob a n)
addBlobs [] = zero
addBlobs (h:xss) = do zs <- createCopy h
                      for_ xss $ \xs -> do
                        liftIO (add [n] xs zs)
                      unsafeFreeze zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    add :: [Int] -> Blob a n -> MBlob a n -> IO ()
    add = memoc memoTable (AddInto (precision @a)) $ \xs zs -> do
      i <- C.get_global_id 0
      at zs i .= at zs i + at xs i

subBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> Blob a n -> m (Blob a n)
subBlob = zipWithBlob' memoTable (Sub (precision @a)) (-)

mulBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> Blob a n -> m (Blob a n)
mulBlob = zipWithBlob' memoTable (Mul (precision @a)) (*)

divideBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> Blob a n -> m (Blob a n)
divideBlob = zipWithBlob' memoTable (Div (precision @a)) (/)

squareBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> m (Blob a n)
squareBlob = mapBlob' memoTable (Square (precision @a)) (^2)

sqrtBlob :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Blob a n -> m (Blob a n)
sqrtBlob = mapBlob' memoTable (Sqrt (precision @a)) sqrt

catBlob :: forall α β m a q. (MonadIO m, KnownNat α, KnownNat β, Storable a) => BlobT q a α -> BlobT q a β -> m (BlobT q a (α + β))
catBlob (Blob xs) (Blob ys) = Blob <$> Buffer.concat [xs, ys]

splitBlob :: forall α β a q. (KnownNat α, KnownNat β) => BlobT q a (α + β) -> (BlobT q a α, BlobT q a β)
splitBlob (Blob xs) = (Blob ys, Blob zs)
  where
    m = fromIntegral $ natVal (Proxy :: Proxy α)
    n = fromIntegral $ natVal (Proxy :: Proxy β)
    ys = Buffer.slice 0 m xs
    zs = Buffer.slice m n xs

mapBlob :: forall n m a k. (MonadIO m, KnownNat n, Storable a,
                            Argument (Array W a), Argument (Array R a), Ord k)
        => KTable k -> k
        -> (Expr a -> CL (Expr a))
        -> Blob a n -> m (Blob a n)
mapBlob table k f = go
  where
    go xs = do
      ys <- createBlob
      liftIO (fKernel [fromIntegral n] xs ys)
      unsafeFreeze ys

    fKernel :: [Int] -> Blob a n -> MBlob a n -> IO ()
    fKernel = memoc table k fSrc
    fSrc :: ArrayR a -> ArrayW a -> CL ()
    fSrc xs ys = do i <- get_global_id 0
                    y <- f (at xs i)
                    at ys i .= y

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)

zipWithBlob :: forall n m a k. (MonadIO m, KnownNat n, Storable a,
                                Argument (Array W a), Argument (Array R a), Ord k)
            => KTable k -> k
            -> (Expr a -> Expr a -> CL (Expr a))
            -> Blob a n -> Blob a n -> m (Blob a n)
zipWithBlob table k f = go
  where
    go xs ys = do
      zs <- createBlob
      liftIO (fKernel [fromIntegral n] xs ys zs)
      unsafeFreeze zs

    fKernel :: [Int] -> Blob a n -> Blob a n -> MBlob a n -> IO ()
    fKernel = memoc table k fSrc
    fSrc :: ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    fSrc xs ys zs = do i <- get_global_id 0
                       z <- f (at xs i) (at ys i)
                       at zs i .= z

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)


mapBlob' :: forall n m a k. (MonadIO m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a), Ord k)
         => KTable k -> k
         -> (Expr a -> Expr a)
         -> Blob a n -> m (Blob a n)
mapBlob' table k f = mapBlob table k (pure . f)

zipWithBlob' :: forall n m a k. (MonadIO m, KnownNat n, Storable a, Argument (Array W a), Argument (Array R a), Ord k)
             => KTable k -> k
             -> (Expr a -> Expr a -> Expr a)
             -> Blob a n -> Blob a n -> m (Blob a n)
zipWithBlob' table k f = zipWithBlob table k (\x y -> pure (f x y))


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
