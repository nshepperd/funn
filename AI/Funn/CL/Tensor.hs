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
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Tensor (
  Tensor(..), MTensor(..),
  KnownDims(..),
  -- Core
  copyInto,
  freeze,
  fromBlob,
  fromList,
  fromVector,
  new,
  replicate,
  thaw,
  toBlob,
  toList,
  toVector,
  unsafeFreeze,
  unsafeThaw,
  reshape,
  reshapeM,
  split,
  append,
  -- Arithmetic
  subTensor,
  mulTensor,
  divTensor,
  mapTensor,
  zipWithTensor
  ) where

import           Control.Applicative
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.IORef
import           Data.List hiding (replicate)
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           GHC.TypeLits
import           Prelude hiding (replicate)
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.Buffer (Buffer)
import qualified AI.Funn.CL.Buffer as Buffer
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MemSub (MemSub)
import qualified AI.Funn.CL.MemSub as MemSub
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space
import qualified Foreign.OpenCL.Bindings as CL

newtype Tensor (ds :: [Nat]) = Tensor { getTensor :: MemSub Double }
newtype MTensor (ds :: [Nat]) = MTensor { getMTensor :: MemSub Double }

-- Classes --

instance Derivable (Tensor ds) where
  type D (Tensor ds) = Tensor ds

instance (MonadIO m, KnownDims ds) => Zero m (Tensor ds) where
  zero = replicate 0

instance (MonadIO m, KnownDims ds) => Semi m (Tensor ds) where
  plus = addTensor

instance (MonadIO m, KnownDims ds) => Additive m (Tensor ds) where
  plusm = addTensors . F.toList

instance (MonadIO m, KnownDims ds) => Scale m Double (Tensor ds) where
  scale = scaleTensor

instance (MonadIO m, KnownDims ds) => VectorSpace m Double (Tensor ds) where
  {}

instance (MonadIO m, KnownDims ds) => Inner m Double (Tensor ds) where
  inner x y = do dot <- mulTensor x y
                 sum <$> toList dot

instance (MonadIO m, KnownDims ds) => Finite m Double (Tensor ds) where
  getBasis b = toList b

instance KnownDims ds => CLType (Tensor ds) (CTensor R ds) where
  karg (Tensor buf) = foldMap karg ds <> MemSub.arg buf
    where
      ds = dimVal (Proxy @ ds)

instance KnownDims ds => CLType (MTensor ds) (CTensor W ds) where
  karg (MTensor buf) = foldMap karg ds <> MemSub.arg buf
    where
      ds = dimVal (Proxy @ ds)

-- Main operations --

new :: forall ds m. (MonadIO m, KnownDims ds) => m (MTensor ds)
new = MTensor <$> MemSub.malloc (product ds)
  where
    ds = dimVal (Proxy :: Proxy ds)

unsafeFreeze :: MTensor ds -> Tensor ds
unsafeFreeze (MTensor mem) = Tensor mem

unsafeThaw :: Tensor ds -> MTensor ds
unsafeThaw (Tensor mem) = MTensor mem

freeze :: (MonadIO m) => MTensor ds -> m (Tensor ds)
freeze (MTensor mem) = Tensor <$> MemSub.clone mem

thaw :: (MonadIO m) => Tensor ds -> m (MTensor ds)
thaw (Tensor mem) = MTensor <$> MemSub.clone mem

fromList :: forall ds m. (MonadIO m, KnownDims ds) => [Double] -> m (Tensor ds)
fromList xs = do when (n /= length xs) $ liftIO $ do
                   throwIO (IndexOutOfBounds "Tensor.fromList")
                 Tensor <$> MemSub.fromList xs
  where
    n = dimSize (Proxy @ ds)

toList :: (MonadIO m) => Tensor ds -> m [Double]
toList (Tensor mem) = MemSub.toList mem

fromVector :: forall ds m. (MonadIO m, KnownDims ds) => S.Vector Double -> m (Tensor ds)
fromVector xs = do when (n /= V.length xs) $ liftIO $ do
                     throwIO (IndexOutOfBounds "Tensor.fromVector")
                   Tensor <$> MemSub.fromVector xs
  where
    n = dimSize (Proxy @ ds)

toVector :: (MonadIO m) => Tensor ds -> m (S.Vector Double)
toVector (Tensor mem) = MemSub.toVector mem

fromBlob :: Blob Double (Prod ds) -> Tensor ds
fromBlob (Blob buf) = Tensor (unsafePerformIO $ Buffer.toMemSub buf)

toBlob :: Tensor ds -> Blob Double (Prod ds)
toBlob (Tensor mem) = Blob (Buffer.fromMemSub mem)

copyInto :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> MTensor ds -> m ()
copyInto (Tensor src) (MTensor dst) = MemSub.copyInto src dst 0 0 (dimSize (Proxy @ ds))

replicate :: forall ds m. (MonadIO m, KnownDims ds) => Double -> m (Tensor ds)
replicate x = fromVector (S.replicate n x)
  where
    n = dimSize (Proxy @ ds)

split :: forall a b. (KnownNat a, KnownNat b)
      => Tensor '[a+b] -> (Tensor '[a], Tensor '[b])
split (Tensor buf) = (Tensor (MemSub.slice 0 a buf),
                      Tensor (MemSub.slice a b buf))
  where
    [a,b] = dimVal (Proxy @ '[a,b])

append :: Tensor '[a] -> Tensor '[b] -> Tensor '[a+b]
append (Tensor one) (Tensor two) = Tensor (unsafePerformIO $ MemSub.concatenate [one, two])

-- Arithmetic operations --

reshape :: (Prod ds ~ Prod es) => Tensor ds -> Tensor es
reshape (Tensor buffer) = Tensor buffer

reshapeM :: (Prod ds ~ Prod es) => MTensor ds -> MTensor es
reshapeM (MTensor buffer) = MTensor buffer

data KName = Add
           | Sub
           | Mul
           | Div
           | Square
           | Sqrt
           | Scale
  deriving (Show, Eq, Ord)

{-# NOINLINE memoTable #-}
memoTable :: KTable KName
memoTable = newKTable unsafePerformIO

scaleTensor :: forall ds m. (MonadIO m, KnownDims ds) => Double -> Tensor ds -> m (Tensor ds)
scaleTensor a xs = do ys <- new
                      liftIO (scale [n] a xs ys)
                      return (unsafeFreeze ys)
  where
    n = dimSize (Proxy @ ds)
    scale :: [Int] -> Double -> Tensor ds -> MTensor ds -> IO ()
    scale = memoc memoTable Scale $ \a (CTensor _ xs) (CTensor _ ys) -> do
      i <- get_global_id 0
      at ys i .= a * at xs i

addTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> Tensor ds -> m (Tensor ds)
addTensor = zipWithTensor memoTable Add (+)

addIntoSrc :: KernelProgram '[TensorCL '[n],
                              MTensorCL '[n]]
addIntoSrc = compile $ \input output -> do
  i <- get_global_id 0
  output![i] .= output![i] + input![i]

addTensors :: forall ds m. (MonadIO m, KnownDims ds) => [Tensor ds] -> m (Tensor ds)
addTensors xss = do out <- unsafeThaw <$> zero
                    for_ xss $ \xs -> do
                      liftIO (clfun addIntoSrc [dimSize out] (reshape xs) (reshapeM out) :: IO ())
                    return (unsafeFreeze out)

subTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> Tensor ds -> m (Tensor ds)
subTensor = zipWithTensor memoTable Sub (-)

mulTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> Tensor ds -> m (Tensor ds)
mulTensor = zipWithTensor memoTable Mul (*)

divTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> Tensor ds -> m (Tensor ds)
divTensor = zipWithTensor memoTable Div (/)

squareTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> m (Tensor ds)
squareTensor = mapTensor memoTable Square (^2)

sqrtTensor :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> m (Tensor ds)
sqrtTensor = mapTensor memoTable Sqrt sqrt

mapTensor :: forall ds m k. (MonadIO m, KnownDims ds, Ord k)
         => KTable k -> k
         -> (Expr Double -> Expr Double)
         -> Tensor ds -> m (Tensor ds)
mapTensor table k f = mapTensorM table k (pure . f)

mapTensorM :: forall ds m k. (MonadIO m, KnownDims ds, Ord k)
           => KTable k -> k
           -> (Expr Double -> CL (Expr Double))
           -> Tensor ds -> m (Tensor ds)
mapTensorM table k f = go
  where
    go xs = do
      ys <- new
      liftIO (fKernel shape xs ys)
      return (unsafeFreeze ys)
    fKernel :: [Int] -> Tensor ds -> MTensor ds -> IO ()
    fKernel = memoc table k fSrc
    fSrc :: TensorCL ds -> MTensorCL ds -> CL ()
    fSrc (CTensor _ xs) (CTensor _ ys) = do
      i <- get_global_id 0
      y <- f (at xs i)
      at ys i .= y
    shape = [dimSize (Proxy @ ds)]

zipWithTensor :: forall ds m k. (MonadIO m, KnownDims ds, Ord k)
              => KTable k -> k
              -> (Expr Double -> Expr Double -> Expr Double)
              -> Tensor ds -> Tensor ds -> m (Tensor ds)
zipWithTensor table k f = zipWithTensorM table k (\x y -> pure (f x y))

zipWithTensorM :: forall ds m k. (MonadIO m, KnownDims ds, Ord k)
               => KTable k -> k
               -> (Expr Double -> Expr Double -> CL (Expr Double))
               -> Tensor ds -> Tensor ds -> m (Tensor ds)
zipWithTensorM table k f = go
  where
    go xs ys = do
      zs <- new
      liftIO (fKernel shape xs ys zs)
      return (unsafeFreeze zs)

    fKernel :: [Int] -> Tensor ds -> Tensor ds -> MTensor ds -> IO ()
    fKernel = memoc table k fSrc
    fSrc :: TensorCL ds -> TensorCL ds -> MTensorCL ds -> CL ()
    fSrc (CTensor _ xs) (CTensor _ ys) (CTensor _ zs) = do
      i <- get_global_id 0
      z <- f (xs ! i) (ys ! i)
      zs!i .= z

    shape :: [Int]
    shape = [dimSize (Proxy @ ds)]

instance (MonadIO m, KnownDims ds) => AdamOps m (Tensor ds) where
  adam_pure_d = replicate
  adam_square_d = squareTensor
  adam_sqrt_d = sqrtTensor
  adam_divide_d = divTensor

instance (MonadIO m, KnownDims ds) => Adam m (Tensor ds) (Tensor ds) where
  adam_update_p = plus
