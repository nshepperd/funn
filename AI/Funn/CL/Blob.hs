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
  Blob, createBlob, freeBlob,
  fromList, toList,
  blobArg,
  pureBlob, scaleBlob, addBlob, subBlob,
  squareBlob, sqrtBlob, divideBlob,
  catBlob, splitBlob,
  mapBlob, zipWithBlob,
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

import           AI.Funn.CL.Code as C

newtype Blob s (n :: Nat) = Blob (Buffer s Float)

-- Classes --

instance Derivable (Blob s n) where
  type D (Blob s n) = Blob s n

instance (MonadCL s m, KnownNat n) => Zero m (Blob s n) where
  zero = pureBlob 0

instance (MonadCL s m, KnownNat n) => Semi m (Blob s n) where
  plus = addBlob

instance (MonadCL s m, KnownNat n) => Additive m (Blob s n) where
  plusm = addBlobs . F.toList

instance (MonadCL s m, KnownNat n) => Scale m Double (Blob s n) where
  scale = scaleBlob

instance (MonadCL s m, KnownNat n) => VectorSpace m Double (Blob s n) where
  {}

instance (MonadCL s m, KnownNat n) => Inner m Double (Blob s n) where
  inner x y = do dot <- mulBlob x y
                 sum <$> toList dot

instance (MonadCL s m, KnownNat n) => Finite m Double (Blob s n) where
  getBasis b = toList b


-- Main operations --

freeBlob :: Blob s n -> OpenCL s ()
freeBlob (Blob mem) = Buffer.free mem

createBlob :: forall n m s. (MonadCL s m, KnownNat n) => m (Blob s n)
createBlob = Blob <$> Buffer.malloc (fromIntegral n)
  where
    n = natVal (Proxy :: Proxy n)

fromList :: forall n m s. (MonadCL s m, KnownNat n) => [Double] -> m (Blob s n)
fromList xs = do when (n /= genericLength xs) $ liftIO $ do
                   throwIO (IndexOutOfBounds "Blob.fromList")
                 Blob <$> Buffer.fromList (map double2Float xs)
  where
    n = natVal (Proxy :: Proxy n)

toList :: (MonadCL s m) => Blob s n -> m [Double]
toList (Blob mem) = map float2Double <$> Buffer.toList mem

blobArg :: Blob s n -> KernelArg s
blobArg (Blob mem) = Buffer.arg mem

-- Arithmetic operations --

pureBlob :: forall n m s. (MonadCL s m, KnownNat n) => Double -> m (Blob s n)
pureBlob x = Blob <$> Buffer.fromVector (S.replicate n (double2Float x))
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)

scaleBlob :: forall n m s. (MonadCL s m, KnownNat n) => Double -> Blob s n -> m (Blob s n)
scaleBlob a xs = do ys <- createBlob
                    runKernel scaleSource "run" [doubleArg a, blobArg xs, blobArg ys] [] [n] [1]
                    return ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    scale :: C.Expr Float -> C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    scale a xs ys = do
      i <- C.get_global_id 0
      C.at ys i .= a * (C.at xs i)
    scaleSource = C.kernel scale

addBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
addBlob = zipWithBlob' (+)

addBlobs :: forall n m s. (MonadCL s m, KnownNat n) => [Blob s n] -> m (Blob s n)
addBlobs [] = zero
addBlobs (Blob one:xss) = do zs <- Blob <$> Buffer.clone one
                             for_ xss $ \xs -> do
                               runKernel addSource "run" [blobArg xs, blobArg zs] [] [n] [1]
                             return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    add :: C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    add xs zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at zs i + C.at xs i
    addSource = C.kernel add

subBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
subBlob = zipWithBlob' (-)

mulBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
mulBlob = zipWithBlob' (*)

divideBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
divideBlob = zipWithBlob' (/)

squareBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> m (Blob s n)
squareBlob = mapBlob' (^2)

sqrtBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> m (Blob s n)
sqrtBlob = mapBlob' sqrt

catBlob :: forall m n s. (KnownNat m, KnownNat n) => Blob s m -> Blob s n -> OpenCL s (Blob s (m + n))
catBlob (Blob xs) (Blob ys) = Blob <$> Buffer.concat [xs, ys]

splitBlob :: forall m n s. (KnownNat m, KnownNat n) => Blob s (m + n) -> (Blob s m, Blob s n)
splitBlob (Blob xs) = (Blob ys, Blob zs)
  where
    m = fromIntegral $ natVal (Proxy :: Proxy m)
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    ys = Buffer.slice 0 m xs
    zs = Buffer.slice m n xs

mapBlob :: forall n m s. (MonadCL s m, KnownNat n) => (Expr Float -> CL (Expr Float)) -> Blob s n -> m (Blob s n)
mapBlob f = go
  where
    go xs = do
      ys <- createBlob
      (runKernel fKernel "run"
       [blobArg xs, blobArg ys]
       [] [fromIntegral n] [])
      return ys

    fKernel = C.kernel fSrc
    fSrc :: ArrayR Float -> ArrayW Float -> CL ()
    fSrc xs ys = do i <- get_global_id 0
                    y <- f (at xs i)
                    at ys i .= y

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)

zipWithBlob :: forall n m s. (MonadCL s m, KnownNat n)
            => (Expr Float -> Expr Float -> CL (Expr Float))
            -> Blob s n -> Blob s n -> m (Blob s n)
zipWithBlob f = go
  where
    go xs ys = do
      zs <- createBlob
      (runKernel fKernel "run"
       [blobArg xs, blobArg ys, blobArg zs]
       [] [fromIntegral n] [])
      return zs

    fKernel = C.kernel fSrc
    fSrc :: ArrayR Float -> ArrayR Float -> ArrayW Float -> CL ()
    fSrc xs ys zs = do i <- get_global_id 0
                       z <- f (at xs i) (at ys i)
                       at zs i .= z

    n :: Int
    n = fromIntegral $ natVal (Proxy :: Proxy n)


mapBlob' :: forall n m s. (MonadCL s m, KnownNat n) => (Expr Float -> Expr Float) -> Blob s n -> m (Blob s n)
mapBlob' f = mapBlob (pure . f)

zipWithBlob' :: forall n m s. (MonadCL s m, KnownNat n)
            => (Expr Float -> Expr Float -> Expr Float)
            -> Blob s n -> Blob s n -> m (Blob s n)
zipWithBlob' f = zipWithBlob (\x y -> pure (f x y))
