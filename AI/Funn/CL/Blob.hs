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
  catBlob, splitBlob
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
addBlob xs ys = do zs <- createBlob
                   runKernel addSource "run" [blobArg xs, blobArg ys, blobArg zs] [] [n] [1]
                   return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    add :: C.ArrayR Float -> C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    add xs ys zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at xs i + C.at ys i
    addSource = C.kernel add

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
subBlob xs ys = do zs <- createBlob
                   runKernel subSource "run" [blobArg xs, blobArg ys, blobArg zs] [] [n] [1]
                   return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    sub :: C.ArrayR Float -> C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    sub xs ys zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at xs i - C.at ys i
    subSource = C.kernel sub

mulBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
mulBlob xs ys = do zs <- createBlob
                   runKernel mulSource "run" [blobArg xs, blobArg ys, blobArg zs] [] [n] [1]
                   return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    mul :: C.ArrayR Float -> C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    mul xs ys zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at xs i * C.at ys i
    mulSource = C.kernel mul

squareBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> m (Blob s n)
squareBlob xs = do ys <- createBlob
                   runKernel squareSource "run" [blobArg xs, blobArg ys] [] [n] [1]
                   return ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    square :: C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    square xs ys = do
      i <- C.get_global_id 0
      C.at ys i .= (C.at xs i)^2
    squareSource = C.kernel square

sqrtBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> m (Blob s n)
sqrtBlob xs = do ys <- createBlob
                 runKernel sqrtSource "run" [blobArg xs, blobArg ys] [] [n] [1]
                 return ys
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    sqrtc :: C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    sqrtc xs ys = do
      i <- C.get_global_id 0
      C.at ys i .= sqrt (C.at xs i)
    sqrtSource = C.kernel sqrtc

divideBlob :: forall n m s. (MonadCL s m, KnownNat n) => Blob s n -> Blob s n -> m (Blob s n)
divideBlob xs ys = do zs <- createBlob
                      runKernel divideSource "run" [blobArg xs, blobArg ys, blobArg zs] [] [n] [1]
                      return zs
  where
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    divide :: C.ArrayR Float -> C.ArrayR Float -> C.ArrayW Float -> C.CL ()
    divide xs ys zs = do
      i <- C.get_global_id 0
      C.at zs i .= C.at xs i / C.at ys i
    divideSource = C.kernel divide

catBlob :: forall m n s. (KnownNat m, KnownNat n) => Blob s m -> Blob s n -> OpenCL s (Blob s (m + n))
catBlob (Blob xs) (Blob ys) = Blob <$> Buffer.concat [xs, ys]

splitBlob :: forall m n s. (KnownNat m, KnownNat n) => Blob s (m + n) -> (Blob s m, Blob s n)
splitBlob (Blob xs) = (Blob ys, Blob zs)
  where
    m = fromIntegral $ natVal (Proxy :: Proxy m)
    n = fromIntegral $ natVal (Proxy :: Proxy n)
    ys = Buffer.slice 0 m xs
    zs = Buffer.slice m n xs
