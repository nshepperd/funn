{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.TensorLazy (
  Tensor(..),
  -- Core
  fromStrict,
  toStrict,
  append,
  nul,
  split,
  reshape,
  appendW,
  splitW
  ) where


import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Blob (Blob, BlobT(Blob))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.LazyMem (LazyMem)
import qualified AI.Funn.CL.LazyMem as LazyMem
import           AI.Funn.CL.MonadCL
import qualified AI.Funn.CL.Tensor as T
import           AI.Funn.Diff.Diff (Derivable(..))
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space

newtype Tensor (ds :: [Nat]) = Tensor { getTensor :: LazyMem Double }

instance Derivable (Tensor ds) where
  type D (Tensor ds) = Tensor ds

instance (MonadIO m, KnownDims ds) => Zero m (Tensor ds) where
  zero = fromStrict <$> zero

instance (MonadIO m, KnownDims ds) => Semi m (Tensor ds) where
  -- Expensive.
  plus a b = fromStrict <$> plus (toStrict a) (toStrict b)

instance (MonadIO m, KnownDims ds) => Additive m (Tensor ds) where
  -- Expensive.
  plusm xs = fromStrict <$> plusm (map toStrict (toList xs))

instance (MonadIO m, KnownDims ds) => Scale m Double (Tensor ds) where
  -- Expensive
  scale x xs = fromStrict <$> scale x (toStrict xs)

instance (MonadIO m, KnownDims ds) => VectorSpace m Double (Tensor ds) where
  {}

instance (MonadIO m, KnownDims ds) => Inner m Double (Tensor ds) where
  -- Expensive.
  inner x y = inner (toStrict x) (toStrict y)

instance (MonadIO m, KnownDims ds) => Finite m Double (Tensor ds) where
  -- Expensive.
  getBasis b = getBasis (toStrict b)

-- O(1)
fromStrict :: T.Tensor ds -> Tensor ds
fromStrict (T.Tensor mem) = Tensor (LazyMem.fromStrict mem)

-- up to O(size)
toStrict :: Tensor ds -> T.Tensor ds
toStrict (Tensor buf) = T.Tensor (unsafePerformIO $ LazyMem.toStrict buf)

-- O(1)
append :: Tensor '[a] -> Tensor '[b] -> Tensor '[a+b]
append (Tensor one) (Tensor two) = Tensor (LazyMem.append one two)

-- O(1)
split :: forall a b. (KnownNat a, KnownNat b) => Tensor '[a+b] -> (Tensor '[a], Tensor '[b])
split (Tensor whole) = (Tensor (LazyMem.slice 0 a whole),
                        Tensor (LazyMem.slice a b whole))
  where
    [a, b] = dimVal (Proxy @[a,b])

-- O(1)
reshape :: (Prod as ~ Prod bs) => Tensor as -> Tensor bs
reshape (Tensor mem) = Tensor mem

nul :: (Prod ds ~ 0) => Tensor ds
nul = Tensor mempty

appendW :: forall ω a b. (KnownDimsF [ω, a, b]) => Tensor '[ω, a] -> Tensor '[ω, b] -> Tensor '[ω, a+b]
appendW (Tensor t1) (Tensor t2) = Tensor $ fold (map part [0..ω-1])
  where
    [ω,a,b] = dimVal (Proxy @[ω, a, b])
    part i = LazyMem.slice (i*a) a t1 <> LazyMem.slice (i*b) b t2

splitW :: forall ω a b. (KnownDimsF [ω, a, b]) => Tensor [ω, a+b] -> (Tensor [ω, a], Tensor [ω, b])
splitW (Tensor whole) = (Tensor (foldMap partA [0..ω-1]),
                         Tensor (foldMap partB [0..ω-1]))
  where
    [ω,a,b] = dimVal (Proxy @[ω, a, b])
    partA i = LazyMem.slice (i*(a+b)) a whole
    partB i = LazyMem.slice (i*(a+b) + a) b whole
