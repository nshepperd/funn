{-# LANGUAGE ConstraintKinds #-}
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
module AI.Funn.CL.Tensor (
  Tensor(..), MTensor(..),
  KnownDims(..),
  -- Core
  copyInto,
  freeze,
  fromList,
  fromVector,
  new,
  replicate,
  thaw,
  toList,
  toVector,
  unsafeFreeze,
  unsafeThaw,
  -- Arithmetic
  reshape,
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
import           Data.Constraint (Dict(..))
import qualified Data.Foldable as F
import           Data.Foldable hiding (toList)
import           Data.IORef
import           Data.List hiding (replicate)
import           Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Foreign.Storable
import           GHC.Float
import           GHC.Stack
import           GHC.TypeLits
import           GHC.Types (Constraint)
import           Prelude hiding (replicate)
import           System.IO.Unsafe

import           AI.Funn.CL.Buffer (Buffer)
import qualified AI.Funn.CL.Buffer as Buffer
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.SGD
import           AI.Funn.Space
import qualified Foreign.OpenCL.Bindings as CL

newtype Tensor (ds :: [Nat]) = Tensor { getTensor :: Buffer Double }
newtype MTensor (ds :: [Nat]) = MTensor { getMTensor :: Buffer Double }

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
  karg (Tensor buf) = foldMap karg ds <> Buffer.arg buf
    where
      ds = dimVal (Proxy @ ds)

instance KnownDims ds => CLType (MTensor ds) (CTensor W ds) where
  karg (MTensor buf) = foldMap karg ds <> Buffer.arg buf
    where
      ds = dimVal (Proxy @ ds)

-- Main operations --

new :: forall ds m. (MonadIO m, KnownDims ds) => m (MTensor ds)
new = MTensor <$> Buffer.malloc (product ds)
  where
    ds = dimVal (Proxy :: Proxy ds)

unsafeFreeze :: MTensor ds -> Tensor ds
unsafeFreeze (MTensor mem) = Tensor mem

unsafeThaw :: Tensor ds -> MTensor ds
unsafeThaw (Tensor mem) = MTensor mem

freeze :: (MonadIO m) => MTensor ds -> m (Tensor ds)
freeze (MTensor mem) = Tensor <$> Buffer.clone mem

thaw :: (MonadIO m) => Tensor ds -> m (MTensor ds)
thaw (Tensor mem) = MTensor <$> Buffer.clone mem

fromList :: forall ds m. (MonadIO m, KnownDims ds) => [Double] -> m (Tensor ds)
fromList xs = do when (n /= length xs) $ liftIO $ do
                   throwIO (IndexOutOfBounds "Tensor.fromList")
                 Tensor <$> Buffer.fromList xs
  where
    n = dimSize (Proxy @ ds)

toList :: (MonadIO m) => Tensor ds -> m [Double]
toList (Tensor mem) = Buffer.toList mem

fromVector :: forall ds m. (MonadIO m, KnownDims ds) => S.Vector Double -> m (Tensor ds)
fromVector xs = do when (n /= V.length xs) $ liftIO $ do
                     throwIO (IndexOutOfBounds "Tensor.fromVector")
                   Tensor <$> Buffer.fromVector xs
  where
    n = dimSize (Proxy @ ds)

toVector :: (MonadIO m) => Tensor ds -> m (S.Vector Double)
toVector (Tensor mem) = Buffer.toVector mem

copyInto :: forall ds m. (MonadIO m, KnownDims ds) => Tensor ds -> MTensor ds -> m ()
copyInto (Tensor src) (MTensor dst) = Buffer.copy src dst 0 0 (dimSize (Proxy @ ds))

replicate :: forall ds m. (MonadIO m, KnownDims ds) => Double -> m (Tensor ds)
replicate x = fromVector (S.replicate n x)
  where
    n = dimSize (Proxy @ ds)

-- Arithmetic operations --

reshape :: (Prod ds ~ Prod es) => Tensor ds -> Tensor es
reshape (Tensor buffer) = Tensor buffer

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

addTensors :: forall ds m. (MonadIO m, KnownDims ds) => [Tensor ds] -> m (Tensor ds)
addTensors xss = do mem <- zero
                    for_ xss $ \(Tensor xs) -> do
                      Buffer.addInto xs (getTensor mem)
                    return mem

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


-- adamTensor :: forall (n :: Nat) m a. (MonadIO m, KnownNat n, CLFloating a, Floats a) => AdamConfig m (Tensor a n) (Tensor a n)
-- adamTensor = defaultAdam {
--   adam_pure_d = pureTensor,
--   adam_scale_d = scale,
--   adam_add_d = plus,
--   adam_square_d = squareTensor,
--   adam_sqrt_d = sqrtTensor,
--   adam_divide_d = divideTensor,
--   adam_update_p = plus
--   }
